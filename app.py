from __future__ import annotations

import base64
import io
import os
import json
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "workspace.db"
SERVICE_SCRIPT = BASE_DIR / "model_service.py"
ALLOWED_MODEL_EXTENSIONS = {".onnx", ".pth", ".pb", ".pt", ".zip"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PORT_MIN = int(os.getenv("MODEL_PORT_MIN", "8000"))
PORT_MAX = int(os.getenv("MODEL_PORT_MAX", "9000"))
BIND_HOST = os.getenv("MODEL_BIND_HOST", "0.0.0.0")
HOST = os.getenv("MODEL_HOST", "")


def default_model_host() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if not HOST:
    HOST = default_model_host()

for directory in (MODELS_DIR, UPLOADS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

state_lock = threading.RLock()
processes: dict[str, subprocess.Popen[Any]] = {}
inference_locks: dict[str, threading.Semaphore] = {}


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                runtime_type TEXT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                class_names TEXT,
                status TEXT NOT NULL,
                host TEXT,
                port INTEGER,
                service_url TEXT,
                pid INTEGER,
                public_key TEXT,
                last_error TEXT,
                last_infer_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(name, version)
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(model_metadata)").fetchall()}
        if "class_names" not in columns:
            conn.execute("ALTER TABLE model_metadata ADD COLUMN class_names TEXT")
        if "runtime_type" not in columns:
            conn.execute("ALTER TABLE model_metadata ADD COLUMN runtime_type TEXT")
            conn.execute("UPDATE model_metadata SET runtime_type = 'yolov5_v5' WHERE runtime_type IS NULL")


init_db()


def json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "message": message}), status


def rows_to_models(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    models = []
    for row in rows:
        model = dict(row)
        model["runtime_type"] = model.get("runtime_type") or "yolov5_v5"
        models.append(model)
    return models


def find_yolov5_repo() -> Path | None:
    candidates = [
        os.getenv("YOLOV5_REPO"),
        str(BASE_DIR / "yolov5"),
        str(BASE_DIR.parent / "model_test" / "yolov5"),
        str(BASE_DIR.parent / "model_test" / "yolov5" / "yolov5-v5.0"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "models" / "experimental.py").exists():
            return path
        for child in path.iterdir() if path.exists() else []:
            if child.is_dir() and (child / "models" / "experimental.py").exists():
                return child.resolve()
    return None


def normalize_class_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def extract_class_names(model_path: Path, algorithm: str, runtime_type: str = "yolov5_v5") -> list[str]:
    if algorithm != "object_detection" or model_path.suffix.lower() != ".pt":
        return []
    if runtime_type == "yolov8_obb":
        return extract_yolov8_class_names(model_path)
    if runtime_type == "yolov5_v6":
        return extract_yolov5_v6_class_names(model_path)
    names = extract_class_names_with_attempt_load(model_path)
    if names:
        return names
    return extract_class_names_from_checkpoint(model_path)


def extract_yolov5_v6_class_names(model_path: Path) -> list[str]:
    repo_path = BASE_DIR / "runtime-type" / "yolov5-6.0"
    if not repo_path.exists():
        return []
    old_cwd = Path.cwd()
    old_model_path = os.environ.get("MODEL_PATH")
    try:
        os.environ["MODEL_PATH"] = str(model_path)
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        os.chdir(repo_path)
        from yolov5 import YOLOv5

        det = YOLOv5()
        names = det.model.module.names if hasattr(det.model, "module") else det.model.names
        return normalize_class_names(names)
    except Exception:
        return []
    finally:
        if old_model_path is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = old_model_path
        os.chdir(old_cwd)


def extract_yolov8_class_names(model_path: Path) -> list[str]:
    repo_path = BASE_DIR / "runtime-type" / "yolov8-obb"
    if not repo_path.exists():
        return []
    try:
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        return normalize_class_names(getattr(model, "names", None))
    except Exception:
        return []


def extract_class_names_with_attempt_load(model_path: Path) -> list[str]:
    repo_path = find_yolov5_repo()
    if repo_path is None:
        return []
    try:
        import torch

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from models.experimental import attempt_load

        model = attempt_load(str(model_path), map_location="cpu")
        names = model.module.names if hasattr(model, "module") else model.names
        return normalize_class_names(names)
    except Exception:
        return []


def extract_class_names_from_checkpoint(model_path: Path) -> list[str]:
    try:
        import torch

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(model_path), map_location="cpu")
    except Exception:
        return []

    names = None
    if isinstance(checkpoint, dict):
        names = checkpoint.get("names")
        for key in ("model", "ema"):
            candidate = checkpoint.get(key)
            if names is None and candidate is not None:
                names = getattr(candidate, "names", None)
    else:
        names = getattr(checkpoint, "names", None)

    return normalize_class_names(names)


def now() -> int:
    return int(time.time())


def model_by_id(model_id: str) -> dict[str, Any] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM model_metadata WHERE id = ?", (model_id,)).fetchone()
    if not row:
        return None
    model = dict(row)
    model["runtime_type"] = model.get("runtime_type") or "yolov5_v5"
    return model


def update_model(model_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [model_id]
    with db() as conn:
        conn.execute(f"UPDATE model_metadata SET {assignments} WHERE id = ?", values)


def used_ports() -> set[int]:
    with db() as conn:
        rows = conn.execute("SELECT port FROM model_metadata WHERE status IN ('DEPLOYING', 'RUNNING') AND port IS NOT NULL").fetchall()
    return {int(row["port"]) for row in rows}


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((HOST, port)) != 0


def allocate_port() -> int:
    occupied = used_ports()
    for port in range(PORT_MIN, PORT_MAX + 1):
        if port not in occupied and is_port_free(port):
            return port
    raise RuntimeError("端口池已耗尽")


def resource_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.05)
    gpu = {"available": False, "used_gb": 0, "total_gb": 0, "percent": 0}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            gpu = {
                "available": True,
                "used_gb": round(used / 1024**3, 2),
                "total_gb": round(total / 1024**3, 2),
                "percent": round(used / total * 100, 1) if total else 0,
            }
    except Exception:
        pass
    return {
        "cpu_percent": cpu,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / 1024**3, 2),
        "memory_total_gb": round(memory.total / 1024**3, 2),
        "gpu": gpu,
    }


def can_start_model() -> tuple[bool, str | None]:
    resources = resource_snapshot()
    if resources["memory_percent"] >= 92:
        return False, "当前服务器算力资源不足，请先关闭其他模型实例"
    return True, None


def health_check(model: dict[str, Any]) -> bool:
    if not model.get("service_url"):
        return False
    try:
        resp = requests.get(f"{model['service_url']}/health", timeout=1.5)
        return resp.ok and resp.json().get("ok") is True
    except requests.RequestException:
        return False


def reconcile_running_models() -> None:
    with db() as conn:
        rows = conn.execute("SELECT * FROM model_metadata WHERE status IN ('DEPLOYING', 'RUNNING')").fetchall()
    for row in rows:
        model = dict(row)
        proc = processes.get(model["id"])
        if proc is not None and proc.poll() is not None:
            update_model(model["id"], status="FAILED_NOT_REACHABLE", port=None, service_url=None, pid=None, last_error="模型进程已退出")
            continue
        healthy = health_check(model)
        if model["status"] == "DEPLOYING" and healthy:
            update_model(model["id"], status="RUNNING", last_error=None)
            inference_locks.setdefault(model["id"], threading.Semaphore(1))
            continue
        if model["status"] == "RUNNING" and not healthy:
            update_model(model["id"], status="FAILED_NOT_REACHABLE", port=None, service_url=None, pid=None, last_error="模型服务健康检查失败")


def wait_until_healthy(service_url: str, timeout_seconds: int = 180) -> tuple[bool, str | None]:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{service_url}/health", timeout=1.5)
            if resp.ok and resp.json().get("ok") is True:
                return True, None
            last_error = resp.json().get("message") or resp.text
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    return False, last_error or "模型启动超时"


def deploy_worker(model_id: str, port: int) -> None:
    model = model_by_id(model_id)
    if not model:
        return
    service_url = f"http://{HOST}:{port}"
    upload_dir = UPLOADS_DIR / model_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SERVICE_SCRIPT),
        "--host",
        BIND_HOST,
        "--port",
        str(port),
        "--model-id",
        model_id,
        "--model-name",
        model["name"],
        "--model-path",
        model["file_path"],
        "--algorithm",
        model["algorithm"],
        "--runtime-type",
        model.get("runtime_type") or "yolov5_v5",
        "--upload-dir",
        str(upload_dir),
    ]
    try:
        log_file = open(LOGS_DIR / f"{model_id}.log", "a", encoding="utf-8")
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start {' '.join(cmd)}\n")
        log_file.flush()
        child_env = os.environ.copy()
        child_env.pop("WERKZEUG_SERVER_FD", None)
        child_env.pop("WERKZEUG_RUN_MAIN", None)
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log_file, stderr=log_file, env=child_env)
        with state_lock:
            processes[model_id] = proc
        update_model(model_id, pid=proc.pid, service_url=service_url)
        healthy, error = wait_until_healthy(service_url)
        if healthy:
            update_model(model_id, status="RUNNING", last_error=None)
            inference_locks[model_id] = threading.Semaphore(1)
            return
        stop_process(model_id)
        update_model(model_id, status="FAILED", port=None, service_url=None, pid=None, last_error=error)
    except Exception as exc:
        stop_process(model_id)
        update_model(model_id, status="FAILED", port=None, service_url=None, pid=None, last_error=str(exc))


def stop_process(model_id: str) -> None:
    with state_lock:
        proc = processes.pop(model_id, None)
    if proc is None:
        model = model_by_id(model_id)
        pid = model.get("pid") if model else None
        if pid:
            try:
                proc_ps = psutil.Process(int(pid))
                proc_ps.terminate()
                proc_ps.wait(timeout=5)
            except Exception:
                pass
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/resources")
def resources():
    return jsonify({"ok": True, "resources": resource_snapshot()})


@app.get("/api/models")
def list_models():
    reconcile_running_models()
    status = request.args.get("status")
    sql = "SELECT * FROM model_metadata"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC"
    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return jsonify({"ok": True, "models": rows_to_models(rows)})


@app.post("/api/models")
def import_model():
    file = request.files.get("file")
    name = request.form.get("name", "").strip()
    version = request.form.get("version", "").strip()
    algorithm = request.form.get("algorithm", "").strip()
    runtime_type = request.form.get("runtime_type", "yolov5_v5").strip() or "yolov5_v5"
    if not file or not file.filename:
        return json_error("请选择模型文件")
    if not name:
        return json_error("模型名称不能为空")
    if not version:
        return json_error("模型版本号不能为空")
    if algorithm not in {"object_detection", "classification", "vlm"}:
        return json_error("算法类型不合法")
    if runtime_type not in {"yolov5_v5", "yolov5_v6", "yolov8_obb"}:
        return json_error("推理后端不合法")
    original = Path(file.filename).name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_MODEL_EXTENSIONS:
        return json_error("仅支持 .onnx、.pth、.pb、.pt、.zip 模型文件")
    model_id = uuid.uuid4().hex
    stored_filename = f"{model_id}{suffix}"
    file_path = MODELS_DIR / stored_filename
    file.save(file_path)
    class_names = extract_class_names(file_path, algorithm, runtime_type)
    class_names_json = json.dumps(class_names, ensure_ascii=False) if class_names else None
    ts = now()
    try:
        with db() as conn:
            conn.execute(
                """
                INSERT INTO model_metadata (
                    id, name, version, algorithm, runtime_type, original_filename, stored_filename, file_path,
                    class_names, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNDEPLOYED', ?, ?)
                """,
                (
                    model_id,
                    name,
                    version,
                    algorithm,
                    runtime_type,
                    original,
                    stored_filename,
                    str(file_path),
                    class_names_json,
                    ts,
                    ts,
                ),
            )
    except sqlite3.IntegrityError:
        file_path.unlink(missing_ok=True)
        return json_error("模型名称和版本号已存在")
    return jsonify({"ok": True, "model": model_by_id(model_id)})


@app.post("/api/models/<model_id>/start")
def start_model(model_id: str):
    with state_lock:
        model = model_by_id(model_id)
        if not model:
            return json_error("模型不存在", 404)
        if model["status"] in {"DEPLOYING", "RUNNING"}:
            return json_error("模型已在部署中或运行中")
        ok, message = can_start_model()
        if not ok:
            return json_error(message or "当前服务器算力资源不足", 400)
        try:
            port = allocate_port()
        except RuntimeError as exc:
            return json_error(str(exc), 400)
        update_model(model_id, status="DEPLOYING", host=HOST, port=port, service_url=f"http://{HOST}:{port}", last_error=None)
        threading.Thread(target=deploy_worker, args=(model_id, port), daemon=True).start()
    return jsonify({"ok": True, "model": model_by_id(model_id)})


@app.post("/api/models/<model_id>/stop")
def stop_model(model_id: str):
    model = model_by_id(model_id)
    if not model:
        return json_error("模型不存在", 404)
    stop_process(model_id)
    inference_locks.pop(model_id, None)
    update_model(model_id, status="STOPPED", port=None, service_url=None, pid=None)
    return jsonify({"ok": True, "model": model_by_id(model_id)})


@app.delete("/api/models/<model_id>")
def delete_model(model_id: str):
    model = model_by_id(model_id)
    if not model:
        return json_error("模型不存在", 404)
    stop_process(model_id)
    inference_locks.pop(model_id, None)
    path = Path(model["file_path"])
    if path.exists() and path.parent.resolve() == MODELS_DIR.resolve():
        path.unlink()
    with db() as conn:
        conn.execute("DELETE FROM model_metadata WHERE id = ?", (model_id,))
    return jsonify({"ok": True})


@app.post("/api/models/<model_id>/publish")
def publish_model(model_id: str):
    model = model_by_id(model_id)
    if not model:
        return json_error("模型不存在", 404)
    if model["status"] != "RUNNING":
        return json_error("只能发布运行中的模型")
    public_key = model["public_key"] or uuid.uuid4().hex[:12]
    update_model(model_id, public_key=public_key)
    return jsonify({"ok": True, "public_url": f"/api/public/{public_key}/predict"})


@app.post("/api/infer")
def infer():
    model_id = request.form.get("model_id", "").strip()
    image = request.files.get("image")
    if not model_id:
        return json_error("请选择运行中的模型")
    if not image or not image.filename:
        return json_error("请上传待测图片")
    model = model_by_id(model_id)
    if not model or model["status"] != "RUNNING" or not model.get("service_url"):
        return json_error("目标模型未运行")
    return proxy_predict(model, image)


@app.post("/infer")
def infer_base64():
    try:
        payload = request.get_json(silent=True) or {}
        image_base64 = str(payload.get("image", "")).strip()
        model_id = str(payload.get("model_id", "")).strip()
        if not image_base64:
            return jsonify({"success": False, "msg": "image is required"})

        model = resolve_infer_model(model_id)
        if model is None:
            return jsonify({"success": False, "msg": "No running object detection model!"})

        image_bytes = decode_base64_image(image_base64)
        image_file = FileStorage(
            stream=io.BytesIO(image_bytes),
            filename="infer.jpg",
            content_type="image/jpeg",
        )
        result = proxy_predict_payload(model, image_file)
        if result.get("ok") is False:
            return jsonify({"success": False, "msg": result.get("message", "Inference failed")})

        formatted = format_legacy_detection_result(result.get("result") or [], model)
        if not formatted["objectVec"]:
            return jsonify({"success": False, "msg": "No detection results!"})
        formatted["success"] = True
        formatted["msg"] = "Detection target output"
        return jsonify(formatted)
    except Exception as exc:
        return jsonify({"success": False, "msg": f"Exception: {exc}"})


@app.post("/api/infer-batch")
def infer_batch():
    model_id = request.form.get("model_id", "").strip()
    files = request.files.getlist("images")
    if not model_id:
        return json_error("请选择运行中的模型")
    if not files:
        return json_error("请上传待测图片")
    model = model_by_id(model_id)
    if not model or model["status"] != "RUNNING" or not model.get("service_url"):
        return json_error("目标模型未运行")

    results = []
    started = time.perf_counter()
    for image in files:
        if not image or not image.filename:
            continue
        item = proxy_predict_payload(model, image)
        item["source"] = Path(image.filename).name
        results.append(item)
    if not results:
        return json_error("没有可推理的图片文件")
    return jsonify(
        {
            "ok": True,
            "model": model,
            "algorithm": model["algorithm"],
            "count": len(results),
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }
    )


@app.post("/api/public/<public_key>/predict")
def public_predict(public_key: str):
    image = request.files.get("image")
    if not image or not image.filename:
        return json_error("请上传 image 文件")
    with db() as conn:
        row = conn.execute("SELECT * FROM model_metadata WHERE public_key = ?", (public_key,)).fetchone()
    if not row:
        return json_error("发布服务不存在", 404)
    model = dict(row)
    if model["status"] != "RUNNING":
        return json_error("发布服务当前不可用", 503)
    return proxy_predict(model, image)


def resolve_infer_model(model_id: str) -> dict[str, Any] | None:
    if model_id:
        model = model_by_id(model_id)
        if model and model["status"] == "RUNNING" and model["algorithm"] == "object_detection" and model.get("service_url"):
            return model
        return None
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM model_metadata WHERE status = 'RUNNING' AND algorithm = 'object_detection' AND service_url IS NOT NULL"
        ).fetchall()
    if len(rows) != 1:
        return None
    return dict(rows[0])


def decode_base64_image(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    return base64.b64decode(image_base64, validate=True)


def format_legacy_detection_result(detections: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    names = model_class_names(model)
    object_vec = []
    for det in detections:
        bbox = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
        label = str(det.get("label") or det.get("class") or "")
        class_id = int(det.get("class_id", class_id_for_label(label, names)))
        item = [
            int(round(float(bbox[0]))),
            int(round(float(bbox[1]))),
            int(round(float(bbox[2]))),
            int(round(float(bbox[3]))),
            float(det.get("score", det.get("confidence", 0))),
            class_id,
        ]
        if is_small_filtered(item, 3):
            continue
        item.append(names[class_id] if 0 <= class_id < len(names) else label)
        keys = ["x0", "y0", "x1", "y1", "confidence", "classId", "className"]
        object_vec.append({"classVec": {key: value for key, value in zip(keys, item)}})
    return {"objectVec": object_vec}


def model_class_names(model: dict[str, Any]) -> list[str]:
    raw = model.get("class_names")
    if not raw:
        return []
    try:
        names = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(name) for name in names] if isinstance(names, list) else []


def class_id_for_label(label: str, names: list[str]) -> int:
    try:
        return names.index(label)
    except ValueError:
        return -1


def is_small_filtered(item: list[Any], class_id: int) -> bool:
    return int(item[5]) == class_id and (int(item[2] - item[0]) < 20 or int(item[3] - item[1]) < 20)


def proxy_predict(model: dict[str, Any], image) -> Any:
    payload = proxy_predict_payload(model, image)
    if payload.get("ok") is False:
        return json_error(payload["message"], int(payload.get("status", 500)))
    return jsonify(
        {
            "ok": True,
            "model": model,
            "algorithm": model["algorithm"],
            "latency_ms": payload["latency_ms"],
            "gateway_latency_ms": payload["gateway_latency_ms"],
            "result": payload["result"],
        }
    )


def proxy_predict_payload(model: dict[str, Any], image) -> dict[str, Any]:
    filename = secure_filename(Path(image.filename).name)
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return {"ok": False, "message": "仅支持 jpg、png、bmp、webp 图片", "status": 400}
    semaphore = inference_locks.setdefault(model["id"], threading.Semaphore(1))
    started = time.perf_counter()
    with semaphore:
        try:
            resp = requests.post(
                f"{model['service_url']}/predict",
                files={"image": (filename, image.stream, image.mimetype or "application/octet-stream")},
                timeout=30,
            )
            data = resp.json()
        except requests.Timeout:
            return {"ok": False, "message": "模型服务响应超时", "status": 504}
        except requests.RequestException as exc:
            update_model(model["id"], status="FAILED_NOT_REACHABLE", port=None, service_url=None, pid=None, last_error=str(exc))
            return {"ok": False, "message": "模型服务不可达", "status": 502}
    gateway_latency = round((time.perf_counter() - started) * 1000, 2)
    if not resp.ok or data.get("ok") is False:
        return {"ok": False, "message": data.get("message", "模型推理失败"), "status": resp.status_code}
    update_model(model["id"], last_infer_at=now())
    return {
        "ok": True,
        "latency_ms": data.get("latency_ms", gateway_latency),
        "gateway_latency_ms": gateway_latency,
        "result": data.get("result"),
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True, threaded=True)

const state = {
  models: [],
  running: [],
  selectedModelFiles: [],
  selectedImages: [],
  imageUrls: [],
  selectedRunningModelId: localStorage.getItem("selectedRunningModelId") || "",
  resultUrls: [],
  resultDownloads: [],
};

const allowedModelPattern = /\.(onnx|pth|pb|pt|zip)$/i;
const allowedImagePattern = /\.(jpg|jpeg|png|bmp|webp)$/i;

const algorithmLabel = {
  object_detection: "目标检测",
  classification: "图像分类",
  vlm: "VLM",
};

const runtimeLabel = {
  yolov5_v5: "YOLOv5-5.0",
  yolov5_v6: "YOLOv5-6.0",
  yolov8_obb: "YOLOv8-OBB",
};

const statusLabel = {
  UNDEPLOYED: "未部署",
  DEPLOYING: "部署中",
  RUNNING: "运行中",
  STOPPED: "已停止",
  FAILED: "部署失败",
  FAILED_NOT_REACHABLE: "不可达",
};

const el = {
  runtimeStatus: document.querySelector("#runtimeStatus"),
  refreshBtn: document.querySelector("#refreshBtn"),
  modelForm: document.querySelector("#modelForm"),
  modelName: document.querySelector("#modelName"),
  modelVersion: document.querySelector("#modelVersion"),
  algorithm: document.querySelector("#algorithm"),
  runtimeType: document.querySelector("#runtimeType"),
  modelFile: document.querySelector("#modelFile"),
  modelFolder: document.querySelector("#modelFolder"),
  selectedModelFiles: document.querySelector("#selectedModelFiles"),
  modelList: document.querySelector("#modelList"),
  modelCount: document.querySelector("#modelCount"),
  cpuMetric: document.querySelector("#cpuMetric"),
  memoryMetric: document.querySelector("#memoryMetric"),
  gpuMetric: document.querySelector("#gpuMetric"),
  runningMetric: document.querySelector("#runningMetric"),
  runningModels: document.querySelector("#runningModels"),
  imageFile: document.querySelector("#imageFile"),
  imageFolder: document.querySelector("#imageFolder"),
  selectedImages: document.querySelector("#selectedImages"),
  previewWrap: document.querySelector("#previewWrap"),
  clearImagesBtn: document.querySelector("#clearImagesBtn"),
  inferBtn: document.querySelector("#inferBtn"),
  downloadAllBtn: document.querySelector("#downloadAllBtn"),
  latencyBadge: document.querySelector("#latencyBadge"),
  resultList: document.querySelector("#resultList"),
  imageViewer: document.querySelector("#imageViewer"),
  viewerImage: document.querySelector("#viewerImage"),
  viewerCaption: document.querySelector("#viewerCaption"),
  viewerClose: document.querySelector("#viewerClose"),
  toast: document.querySelector("#toast"),
};

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => el.toast.classList.add("hidden"), 3600);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) {
    throw new Error(data.message || `请求失败：${response.status}`);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function fileDisplayName(file) {
  return file.webkitRelativePath || file.name;
}

function classNamesText(model) {
  if (!model.class_names) return model.algorithm === "object_detection" ? "类别：未读取到类别名" : "";
  try {
    const names = JSON.parse(model.class_names);
    if (Array.isArray(names) && names.length) return `类别：${names.join("、")}`;
  } catch (error) {
    return `类别：${model.class_names}`;
  }
  return model.algorithm === "object_detection" ? "类别：未读取到类别名" : "";
}

async function refreshAll() {
  const [resources, models] = await Promise.all([
    api("/api/resources"),
    api("/api/models"),
  ]);
  renderResources(resources.resources);
  state.models = models.models;
  state.running = state.models.filter((model) => model.status === "RUNNING");
  renderModels();
  renderRunningSelector();
}

function renderResources(resources) {
  el.cpuMetric.textContent = `${resources.cpu_percent}%`;
  el.memoryMetric.textContent = `${resources.memory_used_gb}G / ${resources.memory_total_gb}G`;
  const gpu = resources.gpu;
  el.gpuMetric.textContent = gpu.available ? `${gpu.used_gb}G / ${gpu.total_gb}G` : "未检测到";
  el.runtimeStatus.textContent = "模型实例关闭网页后仍会保持运行，需显式点击关闭释放端口和资源。";
}

function renderModels() {
  el.modelCount.textContent = state.models.length;
  el.runningMetric.textContent = state.running.length;
  if (!state.models.length) {
    el.modelList.innerHTML = '<div class="empty-block">暂无模型，请先导入模型文件</div>';
    return;
  }
  el.modelList.innerHTML = state.models.map((model) => {
    const running = model.status === "RUNNING";
    const busy = model.status === "DEPLOYING";
    const service = model.service_url ? `<div class="service-url">${escapeHtml(model.service_url)}</div>` : "";
    const error = model.last_error ? `<div class="error-text">${escapeHtml(model.last_error)}</div>` : "";
    const publicUrl = model.public_key ? `<div class="service-url">固定接口：/api/public/${escapeHtml(model.public_key)}/predict</div>` : "";
    const classes = classNamesText(model);
    return `
      <article class="model-item ${running ? "active" : ""} ${model.status.startsWith("FAILED") ? "failed" : ""}">
        <div class="model-main">
          <div>
            <div class="model-name">${escapeHtml(model.name)} <span>${escapeHtml(model.version)}</span></div>
            <div class="model-meta">${algorithmLabel[model.algorithm] || model.algorithm} | ${runtimeLabel[model.runtime_type] || model.runtime_type || "YOLOv5-5.0"} | ${escapeHtml(model.original_filename)}</div>
            ${classes ? `<div class="class-names" title="${escapeHtml(classes)}">${escapeHtml(classes)}</div>` : ""}
            ${service}${publicUrl}${error}
          </div>
          <span class="status ${model.status.toLowerCase()}">${statusLabel[model.status] || model.status}</span>
        </div>
        <div class="model-actions">
          ${running ? `<button class="secondary-button" data-action="stop" data-id="${model.id}">关闭</button>` : `<button class="primary-button" data-action="start" data-id="${model.id}" ${busy ? "disabled" : ""}>${busy ? "启动中" : "启动"}</button>`}
          <button class="ghost-button" data-action="publish" data-id="${model.id}" ${running ? "" : "disabled"}>发布</button>
          <button class="danger-button" data-action="delete" data-id="${model.id}">删除</button>
        </div>
      </article>
    `;
  }).join("");
}

function renderRunningSelector() {
  const current = el.runningModels.value || state.selectedRunningModelId;
  if (!state.running.length) {
    state.selectedRunningModelId = "";
    el.runningModels.innerHTML = '<option value="">暂无运行中模型</option>';
    return;
  }
  el.runningModels.innerHTML = state.running.map((model) => (
    `<option value="${model.id}">${escapeHtml(model.name)}_${escapeHtml(model.version)} (${escapeHtml(model.host)}:${model.port})</option>`
  )).join("");
  const stillRunning = state.running.some((model) => model.id === current);
  const selected = stillRunning ? current : state.running[0].id;
  el.runningModels.value = selected;
  state.selectedRunningModelId = selected;
  localStorage.setItem("selectedRunningModelId", selected);
}

function setModelFiles(files) {
  state.selectedModelFiles = [...files].filter((file) => allowedModelPattern.test(file.name));
  if (!state.selectedModelFiles.length) {
    el.selectedModelFiles.textContent = "尚未选择模型文件";
    return;
  }
  const names = state.selectedModelFiles.slice(0, 6).map(fileDisplayName).join("、");
  const more = state.selectedModelFiles.length > 6 ? ` 等 ${state.selectedModelFiles.length} 个文件` : ` 共 ${state.selectedModelFiles.length} 个文件`;
  el.selectedModelFiles.textContent = `${names}${more}`;
}

async function importModel(event) {
  event.preventDefault();
  const file = state.selectedModelFiles[0] || el.modelFile.files[0];
  if (!file) {
    showToast("请选择模型文件或包含模型文件的文件夹");
    return;
  }
  const form = new FormData();
  form.append("name", el.modelName.value.trim());
  form.append("version", el.modelVersion.value.trim());
  form.append("algorithm", el.algorithm.value);
  form.append("runtime_type", el.runtimeType.value);
  form.append("file", file, file.name);
  await api("/api/models", { method: "POST", body: form });
  el.modelForm.reset();
  state.selectedModelFiles = [];
  el.selectedModelFiles.textContent = "尚未选择模型文件";
  showToast("模型已导入");
  await refreshAll();
}

async function startModel(id) {
  await api(`/api/models/${encodeURIComponent(id)}/start`, { method: "POST" });
  showToast("已提交启动任务");
  await refreshAll();
}

async function stopModel(id) {
  await api(`/api/models/${encodeURIComponent(id)}/stop`, { method: "POST" });
  if (state.selectedRunningModelId === id) {
    state.selectedRunningModelId = "";
    localStorage.removeItem("selectedRunningModelId");
  }
  showToast("模型已关闭");
  await refreshAll();
}

async function publishModel(id) {
  const data = await api(`/api/models/${encodeURIComponent(id)}/publish`, { method: "POST" });
  showToast(`已发布：${data.public_url}`);
  await refreshAll();
}

async function deleteModel(id) {
  const model = state.models.find((item) => item.id === id);
  if (!window.confirm(`确认删除模型 ${model ? `${model.name} ${model.version}` : id}？运行中模型会先关闭。`)) return;
  await api(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (state.selectedRunningModelId === id) {
    state.selectedRunningModelId = "";
    localStorage.removeItem("selectedRunningModelId");
  }
  showToast("模型已删除");
  await refreshAll();
}

function setImages(files) {
  state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
  state.imageUrls = [];
  state.selectedImages = [...files].filter((file) => file.type.startsWith("image/") || allowedImagePattern.test(file.name));
  if (!state.selectedImages.length) {
    el.selectedImages.textContent = "尚未选择图片";
    el.previewWrap.className = "preview-grid empty-preview";
    el.previewWrap.textContent = "暂无图片";
    return;
  }
  const names = state.selectedImages.slice(0, 5).map(fileDisplayName).join("、");
  const more = state.selectedImages.length > 5 ? ` 等 ${state.selectedImages.length} 张` : ` 共 ${state.selectedImages.length} 张`;
  el.selectedImages.textContent = `${names}${more}`;
  renderImagePreview();
}

function renderImagePreview() {
  el.previewWrap.className = "preview-grid";
  el.previewWrap.innerHTML = state.selectedImages.slice(0, 24).map((file, index) => {
    const url = URL.createObjectURL(file);
    state.imageUrls.push(url);
    return `
      <article class="preview-card">
        <button class="image-button" data-action="preview-source" data-src="${url}" data-title="${escapeHtml(fileDisplayName(file))}">
          <img src="${url}" alt="${escapeHtml(fileDisplayName(file))}">
        </button>
        <div class="preview-caption" title="${escapeHtml(fileDisplayName(file))}">${index + 1}. ${escapeHtml(fileDisplayName(file))}</div>
      </article>
    `;
  }).join("");
}

async function runInference() {
  const modelId = el.runningModels.value;
  if (!modelId) {
    showToast("请先启动并选择模型");
    return;
  }
  if (!state.selectedImages.length) {
    showToast("请上传图片或图片文件夹");
    return;
  }
  state.selectedRunningModelId = modelId;
  localStorage.setItem("selectedRunningModelId", modelId);
  el.inferBtn.disabled = true;
  el.inferBtn.textContent = "推理中";
  clearResultUrls();
  const form = new FormData();
  form.append("model_id", modelId);
  state.selectedImages.forEach((file) => form.append("images", file, fileDisplayName(file)));
  try {
    const data = await api("/api/infer-batch", { method: "POST", body: form });
    await renderInferenceBatch(data);
    showToast(`推理完成：${data.count} 张`);
  } finally {
    el.inferBtn.disabled = false;
    el.inferBtn.textContent = "开始推理";
  }
}

async function renderInferenceBatch(data) {
  el.latencyBadge.textContent = `${data.count} 张 / ${data.total_latency_ms} ms`;
  el.latencyBadge.classList.remove("hidden");
  el.downloadAllBtn.classList.remove("hidden");
  el.resultList.className = "result-list";
  el.resultList.innerHTML = "";
  for (let index = 0; index < data.results.length; index += 1) {
    const item = data.results[index];
    const file = state.selectedImages[index];
    const title = item.source || fileDisplayName(file);
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-title-row">
        <div class="result-name" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
        <span class="badge">${item.latency_ms} ms</span>
      </div>
      <div class="result-content">
        <div class="result-media" id="resultMedia${index}"></div>
        <div class="result-body">
          <div class="vlm-text hidden" id="vlmText${index}"></div>
          <pre class="json-tree">${escapeHtml(JSON.stringify(item.result, null, 2))}</pre>
          <div class="result-actions" id="resultActions${index}"></div>
        </div>
      </div>
    `;
    el.resultList.appendChild(card);
    await renderResultMedia(data.algorithm, item, file, index);
  }
}

async function renderResultMedia(algorithm, item, file, index) {
  const media = document.querySelector(`#resultMedia${index}`);
  const actions = document.querySelector(`#resultActions${index}`);
  const vlmText = document.querySelector(`#vlmText${index}`);
  if (!media || !actions) return;
  const imageUrl = URL.createObjectURL(file);
  state.resultUrls.push(imageUrl);
  if (algorithm === "object_detection" && Array.isArray(item.result)) {
    const canvas = await createDetectionCanvas(file, item.result);
    const dataUrl = canvas.toDataURL("image/png");
    const name = downloadName(item.source, "result.png");
    state.resultUrls.push(dataUrl);
    state.resultDownloads.push({ name, url: dataUrl });
    media.innerHTML = "";
    const button = document.createElement("button");
    button.className = "image-button";
    button.dataset.action = "view-result";
    button.dataset.src = dataUrl;
    button.dataset.title = item.source || fileDisplayName(file);
    button.appendChild(canvas);
    media.appendChild(button);
    actions.innerHTML = `<a class="secondary-button" href="${dataUrl}" download="${name}">下载结果图</a>`;
    return;
  }
  if (algorithm === "vlm" && item.result && vlmText) {
    vlmText.textContent = item.result.text || item.result.answer || "";
    vlmText.classList.remove("hidden");
  }
  media.innerHTML = `
    <button class="image-button" data-action="view-result" data-src="${imageUrl}" data-title="${escapeHtml(item.source || fileDisplayName(file))}">
      <img src="${imageUrl}" alt="${escapeHtml(item.source || fileDisplayName(file))}">
    </button>
  `;
  const name = downloadName(item.source, file.name);
  state.resultDownloads.push({ name, url: imageUrl });
  actions.innerHTML = `<a class="secondary-button" href="${imageUrl}" download="${name}">下载原图</a>`;
}

function createDetectionCanvas(file, detections) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    state.resultUrls.push(url);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const context = canvas.getContext("2d");
      context.drawImage(img, 0, 0);
      drawDetections(context, canvas, detections);
      resolve(canvas);
    };
    img.onerror = reject;
    img.src = url;
  });
}

function drawDetections(context, canvas, detections) {
  detections.forEach((det, index) => {
    const [x1, y1, x2, y2] = det.bbox || det.box || [0, 0, 0, 0];
    const label = det.label || det.class || "object";
    const score = Number(det.score ?? det.confidence ?? 0);
    context.strokeStyle = colorFor(index);
    context.lineWidth = Math.max(2, canvas.width / 400);
    if (Array.isArray(det.obb) && det.obb.length >= 4) {
      context.beginPath();
      context.moveTo(det.obb[0][0], det.obb[0][1]);
      det.obb.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
      context.closePath();
      context.stroke();
    } else {
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    const text = `${label} ${(score * 100).toFixed(1)}%`;
    context.font = `${Math.max(14, canvas.width / 48)}px Segoe UI`;
    const metrics = context.measureText(text);
    const labelHeight = Math.max(22, canvas.width / 34);
    context.fillStyle = colorFor(index);
    context.fillRect(x1, Math.max(0, y1 - labelHeight), metrics.width + 12, labelHeight);
    context.fillStyle = "#fff";
    context.fillText(text, x1 + 6, Math.max(16, y1 - 6));
  });
}

function colorFor(index) {
  const colors = ["#0f766e", "#b42318", "#6d5dfc", "#a15c00", "#116a7b", "#14743f"];
  return colors[index % colors.length];
}

function downloadName(source, fallback) {
  const name = String(source || fallback || "result.png").replace(/[\\/:*?"<>|]/g, "_");
  return name.replace(/\.[^.]+$/, "") + "_result.png";
}

function clearResultUrls() {
  state.resultUrls.forEach((url) => {
    if (url.startsWith("blob:")) URL.revokeObjectURL(url);
  });
  state.resultUrls = [];
  state.resultDownloads = [];
  el.downloadAllBtn.classList.add("hidden");
}

function clearImages() {
  state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
  state.imageUrls = [];
  state.selectedImages = [];
  el.imageFile.value = "";
  el.imageFolder.value = "";
  el.selectedImages.textContent = "尚未选择图片";
  el.previewWrap.className = "preview-grid empty-preview";
  el.previewWrap.textContent = "暂无图片";
  clearResultUrls();
  el.latencyBadge.classList.add("hidden");
  el.resultList.className = "result-list empty-block";
  el.resultList.textContent = "暂无推理结果";
}

async function downloadAllResults() {
  if (!state.resultDownloads.length) {
    showToast("暂无可下载的结果");
    return;
  }
  el.downloadAllBtn.disabled = true;
  el.downloadAllBtn.textContent = "打包中";
  try {
    const files = [];
    for (const item of state.resultDownloads) {
      const blob = await fetch(item.url).then((response) => response.blob());
      files.push({ name: item.name, data: new Uint8Array(await blob.arrayBuffer()) });
    }
    const zipBlob = createZip(files);
    const url = URL.createObjectURL(zipBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `inference_results_${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally {
    el.downloadAllBtn.disabled = false;
    el.downloadAllBtn.textContent = "下载全部";
  }
}

function createZip(files) {
  uniqueZipName.seen = new Map();
  const localParts = [];
  const centralParts = [];
  let offset = 0;
  files.forEach((file) => {
    const nameBytes = new TextEncoder().encode(uniqueZipName(file.name, files));
    const crc = crc32(file.data);
    const localHeader = zipHeader(0x04034b50, [
      20, 0x0800, 0, 0, 0, crc, file.data.length, file.data.length, nameBytes.length, 0,
    ], 30);
    localParts.push(localHeader, nameBytes, file.data);
    const centralHeader = zipHeader(0x02014b50, [
      20, 20, 0x0800, 0, 0, 0, crc, file.data.length, file.data.length, nameBytes.length, 0, 0, 0, 0, 0, 0, offset,
    ], 46);
    centralParts.push(centralHeader, nameBytes);
    offset += localHeader.length + nameBytes.length + file.data.length;
  });
  const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
  const end = zipHeader(0x06054b50, [0, 0, files.length, files.length, centralSize, offset, 0], 22);
  return new Blob([...localParts, ...centralParts, end], { type: "application/zip" });
}

function uniqueZipName(name, files) {
  const seen = uniqueZipName.seen || (uniqueZipName.seen = new Map());
  const safe = String(name || "result.png").replace(/[\\/:*?"<>|]/g, "_");
  const count = seen.get(safe) || 0;
  seen.set(safe, count + 1);
  if (!count) return safe;
  return safe.replace(/(\.[^.]+)?$/, `_${count}$1`);
}

function zipHeader(signature, values, length) {
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);
  view.setUint32(0, signature, true);
  const sizes = length === 30 ? [2,2,2,2,2,4,4,4,2,2] : length === 46 ? [2,2,2,2,2,2,4,4,4,2,2,2,2,2,4,4] : [2,2,2,2,4,4,2];
  let cursor = 4;
  values.forEach((value, index) => {
    if (sizes[index] === 4) view.setUint32(cursor, value >>> 0, true);
    else view.setUint16(cursor, value, true);
    cursor += sizes[index];
  });
  return new Uint8Array(buffer);
}

function crc32(data) {
  const table = crc32.table || (crc32.table = Array.from({ length: 256 }, (_, index) => {
    let c = index;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  }));
  let crc = 0xffffffff;
  for (let index = 0; index < data.length; index += 1) {
    crc = table[(crc ^ data[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function openViewer(src, title) {
  el.viewerImage.src = src;
  el.viewerImage.alt = title;
  el.viewerCaption.textContent = title || "";
  el.imageViewer.classList.remove("hidden");
  el.imageViewer.setAttribute("aria-hidden", "false");
}

function closeViewer() {
  el.imageViewer.classList.add("hidden");
  el.imageViewer.setAttribute("aria-hidden", "true");
  el.viewerImage.removeAttribute("src");
}

el.refreshBtn.addEventListener("click", () => refreshAll().catch((error) => showToast(error.message)));
el.modelForm.addEventListener("submit", (event) => importModel(event).catch((error) => showToast(error.message)));
el.modelFile.addEventListener("change", (event) => setModelFiles(event.target.files));
el.modelFolder.addEventListener("change", (event) => setModelFiles(event.target.files));
el.runningModels.addEventListener("change", () => {
  state.selectedRunningModelId = el.runningModels.value;
  localStorage.setItem("selectedRunningModelId", state.selectedRunningModelId);
});
el.imageFile.addEventListener("change", (event) => setImages(event.target.files));
el.imageFolder.addEventListener("change", (event) => setImages(event.target.files));
el.clearImagesBtn.addEventListener("click", clearImages);
el.inferBtn.addEventListener("click", () => runInference().catch((error) => showToast(error.message)));
el.downloadAllBtn.addEventListener("click", () => downloadAllResults().catch((error) => showToast(error.message)));
el.modelList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id } = button.dataset;
  if (action === "start") startModel(id).catch((error) => showToast(error.message));
  if (action === "stop") stopModel(id).catch((error) => showToast(error.message));
  if (action === "publish") publishModel(id).catch((error) => showToast(error.message));
  if (action === "delete") deleteModel(id).catch((error) => showToast(error.message));
});
document.addEventListener("click", (event) => {
  const button = event.target.closest('button[data-action="view-result"], button[data-action="preview-source"]');
  if (button) openViewer(button.dataset.src, button.dataset.title);
});
el.viewerClose.addEventListener("click", closeViewer);
el.imageViewer.addEventListener("click", (event) => {
  if (event.target === el.imageViewer) closeViewer();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el.imageViewer.classList.contains("hidden")) closeViewer();
});

refreshAll().catch((error) => showToast(error.message));
setInterval(() => refreshAll().catch(() => {}), 2000);

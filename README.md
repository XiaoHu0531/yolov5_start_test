# yolov5_start_test

yolov5 系列模型在线分析服务部署。面向测试人员的模型导入、启动、关闭、图片推理与结果可视化工作台。

## 运行

```powershell
cd E:\Code\Model_start_test
pip install -r requirements.txt
python app.py
```

打开 `http://127.0.0.1:5000`。

## 说明

- 模型文件上传后保存到 `models/`，文件名使用 UUID 隔离。
- 模型元数据保存在 `workspace.db`。
- 启动模型会异步拉起一个独立 Flask 推理实例，并分配 `8000-9000` 范围内的端口。
- 目标检测 `.pt` 模型会尝试复用同目录 `yolov5/` 或环境变量 `YOLOV5_REPO` 指向的 YOLOv5 仓库；未配置仓库时返回结构化错误。
- 图像分类和 VLM 目前提供可替换的沙箱推理桩，接口结构已经固定，后续可替换为真实模型加载逻辑。

import sys

from ultralytics import YOLO

import mlflow

sys.path.append("../")
from utils import get_size  # noqa: E402

mlflow.set_tracking_uri("http://localhost:5000")


def load_and_val(model_path: str):
    model = YOLO(model_path, task="detect")
    result = model.val(data="coco128.yaml", imgsz=640)
    return model, result


model = YOLO("yolov8n.pt")

framework = ["onnx", "tflite", "torchscript", "openvino"]
for model_name in framework:
    with mlflow.start_run(run_name=f"yolo {model_name}") as run:
        model_path = model.export(format=model_name)
        exported_model, metrics = load_and_val(model_path)

        mlflow.log_metric("size", get_size(exported_model))

        for key, val in metrics.results_dict.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)
        for key, val in metrics.speed.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)

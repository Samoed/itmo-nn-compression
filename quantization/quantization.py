import sys

from ultralytics import YOLO

sys.path.append("../")
import mlflow
from utils import get_size

mlflow.set_tracking_uri("http://localhost:5000")


def load_and_val(model_path: str):
    model = YOLO(model_path, task="detect")
    result = model.val(data="coco128.yaml", imgsz=640)
    return result


model = YOLO("yolov8n.pt")

params = [("fp32", {}), ("fp16", {"half": True}), ("int8", {"int8": True})]
for model_name, param in params:
    with mlflow.start_run(run_name=f"yolo tflight {model_name}") as run:
        model_path = model.export(format="tflite", **param)
        load_and_val(model_path)

        mlflow.log_metric("size", get_size(model))

        metrics = model.val(data="coco128.yaml")
        for key, val in metrics.results_dict.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)
        for key, val in metrics.speed.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)

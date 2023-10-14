import os

from ultralytics import YOLO

import mlflow

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

        size = 0
        if os.path.isdir(model_path):
            size = sum([os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path)])
        else:
            size = os.path.getsize(model_path)

        mlflow.log_metric("size", size / (2**20))

        for key, val in metrics.results_dict.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)
        for key, val in metrics.speed.items():
            new_key = key.replace("(", "").replace(")", "")
            mlflow.log_metric(new_key, val)

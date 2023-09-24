import sys

from ultralytics import YOLO

sys.path.append("../")
import mlflow  # ignore: E402
from utils import get_size

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name="yolo run ultralytics") as run:
    model = YOLO("yolov8n.pt")
    mlflow.log_metric("size", get_size(model))

    metrics = model.val(data="coco128.yaml")

    for key, val in metrics.results_dict.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)
    for key, val in metrics.speed.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)
    # mlflow.log_figure(metrics.confusion_matrix, 'conf_matrix.png')

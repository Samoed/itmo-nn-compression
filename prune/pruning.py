from ultralytics import YOLO
import torch
from torch.nn.utils import prune
import sys

sys.path.append("../")
from utils import get_size
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")


with mlflow.start_run(run_name="yolo prune .3 params") as run:
    model = YOLO("yolov8n.pt")

    zeros = 0
    n_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.3)
            prune.remove(module, "weight")
            zeros += float(torch.sum(module.weight == 0))
            n_elements += float(module.weight.nelement())
    mlflow.log_metric("size", get_size(model))
    mlflow.log_metric("sparsity", 100.0 * zeros / n_elements)

    metrics = model.val(data="coco128.yaml")
    for key, val in metrics.results_dict.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)
    for key, val in metrics.speed.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)
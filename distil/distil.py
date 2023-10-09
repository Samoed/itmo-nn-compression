import sys

import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm.autonotebook import tqdm
from ultralytics import YOLO

sys.path.append("../")
from utils import get_size  # noqa: E402

mlflow.set_tracking_uri("http://localhost:5000")
dataset_path = "../../../datasets/data/coco"
COCO_PATH = f"{dataset_path}/images/val2017"
COCO_ANN_PATH = f"{dataset_path}/annotations/instances_val2017.json"
TEACHER_MODEL = "yolov8m.pt"
STUDENT_MODEL = "yolov8n.pt"
SAVE_PATH = "distilled_yolo.pt"
T = 1
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

with mlflow.start_run(run_name="yolo distil params") as run:
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    coco_dataset = CocoDetection(root=COCO_PATH, annFile=COCO_ANN_PATH, transform=data_transform)
    dataset_size = len(coco_dataset)

    train_split = int(0.8 * dataset_size)
    indices = list(range(dataset_size))
    train_sampler = SubsetRandomSampler(indices[:train_split])
    val_sampler = SubsetRandomSampler(indices[train_split:])

    train_loader = DataLoader(coco_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=lambda x: x)
    val_loader = DataLoader(coco_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=lambda x: x)

    # download yolo models
    teacher_model_yolo = YOLO(TEACHER_MODEL)
    student_model_yolo = YOLO(STUDENT_MODEL)
    del teacher_model_yolo, student_model_yolo

    device = torch.device("cuda")  # conv2d halve not implemented for gpu

    teacher_model = torch.load(TEACHER_MODEL)["model"].to(device)
    student_model = torch.load(STUDENT_MODEL)["model"].to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = torch.stack([i for i, _ in inputs]).to(device).half()

            teacher_outputs = teacher_model(inputs)[0]
            student_outputs = student_model(inputs)[0]

            loss_kd = T**2 * criterion(F.softmax(student_outputs / T, dim=-1), F.softmax(teacher_outputs / T, dim=-1))
            loss_kd.requires_grad = True

            optimizer.zero_grad()
            loss_kd.backward()
            optimizer.step()

            running_loss += loss_kd.item()

        # Print training loss for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    mlflow.log_metric("size", get_size(student_model))

    torch.save({"model": student_model}, SAVE_PATH)
    model = YOLO(SAVE_PATH, task="detect")
    metrics = model.val(data="coco128.yaml")
    for key, val in metrics.results_dict.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)
    for key, val in metrics.speed.items():
        new_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(new_key, val)

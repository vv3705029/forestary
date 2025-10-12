# src/deforestation/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def load_deforestation_model(model_path, device):
    # Use latest torchvision format (no warnings)
    model = resnet50(weights=None)  # pretrained=False equivalent
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)  # 2 classes: Non-Deforested / Deforested
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

import torch
import torch.nn as nn

MODEL_PATH1 = "models/wildfire_model.pth"

class WildfireNet(nn.Module):
    """
    Model structure must match your training.
    """
    def __init__(self, input_dim):
        super(WildfireNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)  # 2 classes: fire/no fire

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def load_model(input_dim):
    """
    Load PyTorch model from MODEL_PATH1
    """
    model = WildfireNet(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH1, map_location=torch.device('cpu')))
    model.eval()
    return model

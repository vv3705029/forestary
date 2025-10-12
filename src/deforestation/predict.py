# src/deforestation/predict.py
import torch
from torchvision import transforms
from PIL import Image

def predict_deforestation(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        print(probs)
        pred_class = torch.argmax(probs, dim=1).item()

    class_names = ["Deforested","Non-Deforested"]
    confidence = float(probs[0][pred_class])
    
    return class_names[pred_class], confidence

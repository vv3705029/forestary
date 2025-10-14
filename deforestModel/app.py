import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load Model
# ---------------------------
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model.load_state_dict(torch.load("deforestation_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------------------
# Class names
# ---------------------------
class_names = ['Deforested', 'Non-deforested']

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

def predict_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.cpu().numpy()

@app.get("/")
def root():
    return {"message": "Welcome to the Image Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # Read image bytes
        pred_class, probs = predict_image_from_bytes(contents)
        predicted_label = class_names[pred_class]
        probs_list = probs.flatten().tolist()
        return JSONResponse(content={
            "predicted_class": predicted_label,
            "class_index": pred_class,
            "probabilities": probs_list
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

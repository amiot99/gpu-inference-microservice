from fastapi import FastAPI
from fastapi import UploadFile, File, Form
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import io

with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_cache = {
    "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval(),
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
}
app = FastAPI()

@app.get("/",response_class=HTMLResponse) #this tells FastAPI that the function right below is in charge of handling requests that go to:
#the path /, using a get operation
async def root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.post("/predict")
async def predict(model: str = Form(...),image: UploadFile = File(...)):
    if model not in model_cache:
        raise HTTPException(status_code=400, detail="Unsupported model")
    contents = await image.read()
    convert_contents = Image.open(io.BytesIO(contents)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(convert_contents)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)

    selected_model = model_cache[model]
    with torch.no_grad():
        outputs = selected_model(image_tensor)

    predicted_idx = outputs.argmax(dim=1).item()
    label = labels[predicted_idx]
    return {
        "filename": image.filename,
        "predicted_index": predicted_idx,
        "predicted_label": label,
        "model_used" : model
    }

#Now that I have an image loaded in and set to contents I need to convert it into a usable
#format that the model can understand





from fastapi import FastAPI
from fastapi import UploadFile, File
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
my_model = models.resnet18(pretrained=True)
my_model = my_model.to(device)
my_model.eval()

app = FastAPI()

@app.get("/") #this tells FastAPI that the function right below is in charge of handling requests that go to:
#the path /, using a get operation
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
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
    with torch.no_grad():
        outputs = my_model(image_tensor)

    predicted_idx = outputs.argmax(dim=1).item()
    label = labels[predicted_idx]
    return {
        "filename": image.filename,
        "predicted_index": predicted_idx,
        "predicted_label": label
    }

#Now that I have an image loaded in and set to contents I need to convert it into a usable
#format that the model can understand





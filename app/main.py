import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
from typing import List


app = FastAPI()

data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model_path = '../model/similarity_model_1_views_part_colors_cut_segment.pth'
model = models.squeezenet1_1(pretrained=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


def load_img(image_data: bytes, img_transforms: transforms = data_transforms) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_data))
    tensor_image = img_transforms(image)
    return tensor_image.unsqueeze(0)

def vectorize_image(image_data: bytes) -> np.array:
    tensor_image = load_img(image_data, data_transforms)
    with torch.no_grad():
        tensor_image = tensor_image
        embedding = model.features[:](tensor_image.to(device)).sum(2).sum(2)
    return embedding.cpu().numpy()


class Images(BaseModel):
    image: bytes


@app.get('/')
def test():
    return {'Status': 'Ok'}


@app.post('/vectorize')
def vectorize(image_data: UploadFile = File(...)):
    embedding = vectorize_image(image_data.file.read())
    result = {
        image_data.filename: embedding.tolist()
    }    
    return result


@app.post('/vectorize batch')
def vectorize_images(images_data: List[UploadFile] = File(...)) -> dict:
    embeddings = {}
    for image_data in images_data:
        name = image_data.filename
        embeddings[name] = vectorize_image(image_data.file.read()).tolist()
    return embeddings


#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)

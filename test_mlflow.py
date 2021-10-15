import torch
import mlflow.pytorch
from torchvision import models, transforms
import torch
import torchvision

from PIL import Image

#preprocess image input using function transforms.compose from torchvision library
#The input data needs to be resized to (224,224), and normalized to fit the model input
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#Labels mapping
class_name = ["mask", "unmask"]

model_uri = "model"
loaded_model = mlflow.pytorch.load_model(model_uri)

img_path = "/hdd/tuannm82/face-mask-classifier/test1.jpg"
input_image = Image.open(img_path)
img = preprocess(input_image)
img = img.unsqueeze(0)
img = img.to('cuda')
print(loaded_model)

predictions = loaded_model(img)
softmax = torch.nn.functional.softmax(predictions[0], dim=0).to('cpu')
index = torch.argmax(softmax, dim=0)
labels_name = class_name[index]
probabilities = "{:.2f}%".format(softmax[index].item()*100)
print("Label: {}, probability: {}".format(labels_name, probabilities))

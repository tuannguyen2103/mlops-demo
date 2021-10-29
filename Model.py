import torch
import mlflow.pytorch
from torchvision import models, transforms
import torch
import torchvision
import numpy as np
import cv2

from PIL import Image

class Model(object):
    def __init__(self):
        print("Initializing")
        model_uri = "model"
        self.trained_model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
        self.trained_model.eval()
    
    def pre_processing(self, img: np.ndarray):
        preprocess_input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        print("img: ", img.shape)
        print("img shape", img.shape)
        print("type: ", type(img))
            # cv2.imshow("img", img)
        img = cv2.resize(img.astype(np.float32), (224,224))
        print("img shape", img.shape)
        img = preprocess_input(img)
        img = img.unsqueeze(0)
        return img

    def predict(self, X,features_names):
        print("Predict called")
        print(X)
        print(type(X))
        print("X: ", X.shape)
        input_img = self.pre_processing(X)
        class_name = ["mask", "unmask"]
        predictions = self.trained_model(input_img)
        softmax = torch.nn.functional.softmax(predictions[0], dim=0).to('cpu')
        index = torch.argmax(softmax, dim=0)
        labels_name = class_name[index]
        probabilities = "{:.2f}%".format(softmax[index].item()*100)
        print("Label: {}, probability: {}".format(labels_name, probabilities))
        return [labels_name, probabilities]
        # return self.trained_model(input_img)

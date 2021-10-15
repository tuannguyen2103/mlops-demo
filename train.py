from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision

import mlflow.pytorch

import time
import os
import copy

#Dictionary of data transform
data_transforms = {
    #Composition of transformation
    'train': transforms.Compose([
        #resize the data image to the given size
        transforms.Resize((224,224)),
        #flip the data images horizontally with a probability, if empty default is p = 0.5
        transforms.RandomHorizontalFlip(),
        #turn the data from numpy'array to tensor
        transforms.ToTensor(),
        #Normalize a tensor image with mean and standard deviation
        # 3 channels - 3 means and 3 standard devivation
        #formula: output[channel] = (input[channel] - mean[channel]) / std[channel]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#set path to dataset, in this case using FMLD dataset
data_dir = './dataset/'
#transform data from folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# load data after transformation      
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
# find the size of the dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# find all the classes names in the dataset
class_names = image_datasets['train'].classes
# set the device(CUDA if availble or else use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    # initialize the timestamp
    since = time.time()
    # create a deep copy of the model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                metrics = {"val acc": np.float64(epoch_acc), "val loss": np.float64(epoch_loss)}


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

## Finetune the convNet

model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model, metrics = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "mask-detector", registered_model_name="MLops-demo2")
    mlflow.log_params({"epoch": 5, "optimizer": "SGD, lr=0.0001, momentum=0.9", "scheduler": "StepLR,step_size=5, gamma=0.1", "loss": "CrossEntropyLoss"})

    mlflow.log_metrics(metrics)
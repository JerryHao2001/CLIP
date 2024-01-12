import os
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import Food101, OxfordIIITPet, StanfordCars, Flowers102
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
root = "./data"


weights = ResNet18_Weights.DEFAULT
preprocess2 = weights.transforms()
model = resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 100)

train = Flowers102(root, download=True, transform=preprocess2)
test = Flowers102(root, download=True, split = 'test', transform=preprocess2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)

for epoch in range(5):
    for images, target in tqdm(DataLoader(train, 32)):
        outputs = model(images.to(device))
        loss = criterion(outputs, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
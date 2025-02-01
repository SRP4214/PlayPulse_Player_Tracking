import requests
import cv2
import json
import numpy as np
import os
from io import BytesIO
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 21)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.fc1(x))
        out = self.fc2(x)
        
        return out

# model = torch.load(r"C:\Users\Dell\Downloads\weapon_detect.pth", map_location=torch.device('cpu'))
# model.eval()
def check_gun(cropped_image, model):
    with open('ValorantWeaponData.json', 'r') as f:
        weapon_data = json.load(f)
    _, w, _ = cropped_image.shape
    gun_crop = cropped_image[:, int(0.33 * w):int(0.66 * w)]
    # cv2.imwrite('gun_crop.jpg', gun_crop)
    with torch.no_grad():
        img = Image.fromarray(gun_crop)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        pred = model(img_tensor)
        x = (torch.argmax(pred, dim=1)).item()
        return weapon_data[x]['displayName']

# img = cv2.imread('output_image2.jpg')
# check_gun(img, weapon_model)

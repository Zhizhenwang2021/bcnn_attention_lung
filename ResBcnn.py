import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, sampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, models
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import xml.etree.ElementTree as ET
import torch
import torchvision
from tqdm import tqdm
from resnet import resnet50

# model_path = "E:\\checkpoint.pth.tar"
from torch import FloatTensor

classnumber = 10

def new_parameter(*size):
    out = nn.Parameter(FloatTensor(*size), requires_grad=True)
    torch.nn.init.xavier_normal_(out)
    return out

class Attention(nn.Module):

    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x

features = 2048
fmap_size = 7
os.environ['CUDA_VISIBLE_DEVICES'] =','.join(str(x) for x in [0,])
class BCNN(nn.Module):
    def __init__(self, fine_tune=False):
        super(BCNN, self).__init__()
        # checkpoint = torch.load(model_path)
        resnet = resnet50(pretrained=False)
        # resnet.load_state_dict(checkpoint['state_dict'])
        for param in resnet.parameters():
                param.requires_grad = True
        layers = list(resnet.children())[:-2]
        self.features = nn.Sequential(*layers)
        self.attn = Attention(2048)
        self.fc = nn.Linear(features , classnumber)
        self.dropout = nn.Dropout(0.5)
        nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size()[0]
        x = self.features(x)
        x = x.view(N, features, fmap_size ** 2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (fmap_size ** 2)
        x = torch.sqrt(x + 1e-5)
        x = self.attn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

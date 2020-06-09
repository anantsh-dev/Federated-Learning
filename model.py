#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script used to define the Model
@author : Anant
"""
import torch.nn as nn
import torch.nn.functional as F

#Model
#Image (1x28x28)
#Conv Layer 1 (32x28x28) + Relu Activation 
#MaxPool (32x14x14)
#Conv Layer 2 (64x14x14) + Relu Activation
#MaxPool (64x7x7)
#FC layer 1 (3136) + Relu Activation
#FC Layer 2 (128)
#Output Layer (10)

#Final image size before FC layer
FLATTEN_SIZE = 64*7*7


class CNN(nn.Module):
    def __init__(self):
        """
        Initializes the CNN Model Class and the required layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(FLATTEN_SIZE, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Form the Feed Forward Network by combininig all the layers
        :param x: the input image for the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        pred = F.log_softmax(x, dim=1)
        return pred
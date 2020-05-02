# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:52:53 2020

@author: sss
"""

import torch
import torch.nn as nn

# defining a pytorch class
class Classify(nn.Module):
    
    def __init__(self , in_feature , hidden_1 , hidden_2 , out_feature):
        super (Classify, self).__init__() 
        
        self.in_feature = in_feature
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.out_feature = out_feature
        
        # framing the layers
        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(in_feature, hidden_1, kernel_size=(5, 5))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden_1, hidden_2, kernel_size=(5, 5))
        self.fc1 = nn.Linear(hidden_2 * 75 * 75, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_feature)

    def forward(self, image):
        
        # setting the flow of input images
        x = self.pool(self.conv1(self.pad(image)))
        x = self.pool(self.conv2(self.pad(x)))
        x = x.view(-1, self.hidden_2 * 75 * 75)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
        
        
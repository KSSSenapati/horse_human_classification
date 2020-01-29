# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:02:14 2020

@author: sss
"""

import torch
import torchvision
import torchvision.transforms as transforms

def getdata(train_path, test_path):
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset = torchvision.datasets.ImageFolder(root = "data/train", transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = True, num_workers = 4)
    
    testset = torchvision.datasets.ImageFolder(root = "data/train", transform = transform)
    trainloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True, num_workers = 4)

    return trainloader, testloader, trainset.classes

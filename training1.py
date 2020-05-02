# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:32:08 2020

@author: Asus
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from model import Classify
from dataset import getdata

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

trainloader, testloader, classes = getdata('./data/train', './data/validation')
    
# get rand image
dataiter = iter(trainloader)
images, labels = dataiter.next()


# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels]))

model = Classify(in_feature=3, out_feature=2, hidden_1=6, hidden_2=16)

model = model.cuda()

model.train()

################################## training started ###########################

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

training_loss = []
min_loss = 100
val_loss = []


for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data[0].cuda(), data[1].cuda()
        #output = model(images)
        
        #zero the parameters in the gradient
        optimiser.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        training_loss.append(loss.item())
        
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        
        if i % 10 == 9 :
            print('[%d, %5d] training loss : %0.3f' %(epoch+1, i+1, running_loss / 10))
            
            running_loss = 0.0

print ("training completed")
    # validation done at the same time 
    '''with torch.no_grad():
        model.eval()
        
        val_loss = 0.0
        
        for i, data in enumerate(testloader, 0):
            val_images, val_labels = data[0].cuda, data[1].cuda()
            
            val_output = model(val_images)
            
            loss = criterion(val_output, val_labels)
            val_loss.append(loss.item())
            
            if loss.item()<min_loss:
                min_loss = loss.item()
                
                print('model saved with improvements')
                
                torch.save(model.state_dict(), 'trained_weights_' + 
                           'epoch_' + str(epoch+1) + 'loss_' + str(val_loss) + '.pt')
                
            else:
                print('no improvements')'''
   
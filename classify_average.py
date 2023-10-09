# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:49:06 2023

@author: ouyangg
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import DataLoader
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib
import importlib as imp
import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
from sklearn.utils import shuffle
import os
import copy
import mat73
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


#%% common parameters:
batch_size = 50
epochs = 200
EEG_PATH = 'ave_cwt_erp_same_trial.mat'
data_mat = loadmat(EEG_PATH)
label= data_mat['label'].squeeze(0)
idx_train_test = np.squeeze(loadmat('rand400.mat')['temp']) - 1
freq_idx = np.random.randint(2,7,size=200)
idx = np.concatenate((np.array(range(6,1201,6))-1,np.array(range(6,1201,6))-freq_idx))
# idx = np.concatenate((np.array(range(6,1201,6))-np.random.randint(2,7),np.array(range(6,1201,6))-1))
label1 = label[idx]
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4

data_erp = np.transpose(data_mat['erp_all'], (2,0,1))
data_erp = data_erp[idx,:,:]
data_cwta = np.transpose(data_mat['cwt_a_all'], (2,0,1))
data_cwta = data_cwta[idx,:,:]
data_cwtp = np.transpose(data_mat['cwt_p_all'], (2,0,1))
data_cwtp = data_cwtp[idx,:,:]

random.shuffle(idx_train_test)
train_data_erp = np.array(data_erp[idx_train_test[0:300],:,:])
test_data_erp = np.array(data_erp[idx_train_test[300:400],:,:])
train_data_cwta = np.array(data_cwta[idx_train_test[0:300],:,:])
test_data_cwta = np.array(data_cwta[idx_train_test[300:400],:,:])
train_data_cwtp = np.array(data_cwtp[idx_train_test[0:300],:,:])
test_data_cwtp = np.array(data_cwtp[idx_train_test[300:400],:,:])
train_label = np.array(label1[idx_train_test[0:300]])
test_label = np.array(label1[idx_train_test[300:400]])

#%% common classes
class EEGdataset1(Dataset):
    def __init__(self,feature,label):
        self.feature = feature
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        sample = self.feature[idx]
        label = self.label[idx]
        return sample, label

class EEGdataset2(Dataset):
    def __init__(self,feature1,feature2,label):
        self.feature1 = feature1
        self.feature2 = feature2
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        sample1 = self.feature1[idx]
        sample2 = self.feature2[idx]
        label = self.label[idx]
        return sample1, sample2, label
    
class EEGdataset3(Dataset):
    def __init__(self,feature1,feature2,feature3,label):
        self.feature1 = feature1
        self.feature2 = feature2
        self.feature3 = feature3
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        sample1 = self.feature1[idx]
        sample2 = self.feature2[idx]
        sample3 = self.feature3[idx]
        label = self.label[idx]
        return sample1, sample2, sample3, label

class MyNN1(nn.Module):
    def __init__(self,input_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_model = nn.Sequential(
            nn.Linear(input_len, 2),
            # nn.LayerNorm(2),
            nn.BatchNorm1d(2),
            )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_model(x.float())
        # logits = logits - (torch.mean(logits,dim=1,keepdim=True)).repeat(1,2)
        return logits
    
class MyNN_cb2(nn.Module):
    def __init__(self, Model1, Model2):
        super().__init__()
        self.Model1 = Model1
        self.Model2 = Model2
        self.classifier = nn.Sequential(
            nn.Linear(4,2),
            )
    def forward(self, x1, x2):
        x1 = self.Model1(x1)
        x2 = self.Model2(x2)
        x = torch.cat((x1,x2),dim=1)
        x = F.softmax(self.classifier(x))
        return x

class MyNN_cb3(nn.Module):
    def __init__(self, Model1, Model2, Model3):
        super().__init__()
        self.Model1 = Model1
        self.Model2 = Model2
        self.Model3 = Model3
        self.classifier = nn.Sequential(
            nn.Linear(6,2),
            )
    def forward(self, x1, x2, x3):
        x1 = self.Model1(x1)
        x2 = self.Model2(x2)
        x3 = self.Model3(x3)
        x = torch.cat((x1,x2,x3),dim=1)
        x = F.softmax(self.classifier(x))
        return x


#%% common class for 1D
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = F.softmax(model(X))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = F.softmax(model(X))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def train_loop2(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X1, X2, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X1, X2)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop2(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X1, X2, y in dataloader:
            pred = model(X1, X2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def train_loop3(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X1, X2, X3, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X1, X2, X3)
        loss = loss_fn(pred, y)
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    return 100*correct
            
def test_loop3(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X1, X2, X3, y in dataloader:
            pred = model(X1, X2, X3)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct


#%% train ERP only
trainset = EEGdataset1(train_data_erp, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset1(test_data_erp, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)

model_erp = MyNN1(train_data_erp.shape[1]*train_data_erp.shape[2])
optimizer = torch.optim.SGD(model_erp.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print(scheduler.get_last_lr()[0])
    train_loop(train_loader, model_erp, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop(test_loader, model_erp, loss_fn)
    accs.append(tmp)
print("Done!")
accs_erp = accs

#%% train cwta only
trainset = EEGdataset1(train_data_cwta, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset1(test_data_cwta, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)

model_cwta = MyNN1(train_data_cwta.shape[1]*train_data_cwta.shape[2])
optimizer = torch.optim.SGD(model_cwta.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model_cwta, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop(test_loader, model_cwta, loss_fn)
    accs.append(tmp)
print("Done!")
accs_cwta = accs

#%% train cwtp only
trainset = EEGdataset1(train_data_cwtp, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset1(test_data_cwtp, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)

model_cwtp = MyNN1(train_data_cwtp.shape[1]*train_data_cwtp.shape[2])
optimizer = torch.optim.SGD(model_cwtp.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model_cwtp, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop(test_loader, model_cwtp, loss_fn)
    accs.append(tmp)
print("Done!")
accs_cwtp = accs

#%% train ERP and cwta only
trainset = EEGdataset2(train_data_erp, train_data_cwta, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset2(test_data_erp, test_data_cwta, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)

# temp11 = MyNN1(train_data_erp.shape[1]*train_data_erp.shape[2])
# temp22 = MyNN1(train_data_cwta.shape[1]*train_data_cwta.shape[2])
# model11 = copy.deepcopy(temp11)
# model22 = copy.deepcopy(temp22)
model11 = copy.deepcopy(model_erp)
model22 = copy.deepcopy(model_cwta)
model_erp_cwta = MyNN_cb2(model11,model22)
model_erp_cwta.eval()

for name, param in model_erp_cwta.named_parameters():
    print(name)
    print(param)
    if name=='classifier.0.weight':
        param.data = nn.parameter.Parameter(torch.ones_like(param))
    if name=='classifier.0.bias':
        param.data = nn.parameter.Parameter(torch.zeros_like(param))

optimizer = torch.optim.SGD(model_erp_cwta.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop2(train_loader, model_erp_cwta, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop2(test_loader, model_erp_cwta, loss_fn)
    accs.append(tmp)
print("Done!")
accs_erp_cwta = accs

#%% train ERP and cwtp only
trainset = EEGdataset2(train_data_erp, train_data_cwtp, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset2(test_data_erp, test_data_cwtp, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)


model11 = copy.deepcopy(model_erp)
model22 = copy.deepcopy(model_cwtp)
model_erp_cwtp = MyNN_cb2(model11,model22)
model_erp_cwtp.eval()

for name, param in model_erp_cwtp.named_parameters():
    print(name)
    print(param)
    if name=='classifier.0.weight':
        param.data = nn.parameter.Parameter(torch.ones_like(param))
    if name=='classifier.0.bias':
        param.data = nn.parameter.Parameter(torch.zeros_like(param))

optimizer = torch.optim.SGD(model_erp_cwtp.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop2(train_loader, model_erp_cwtp, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop2(test_loader, model_erp_cwtp, loss_fn)
    accs.append(tmp)
print("Done!")
accs_erp_cwtp = accs

#%% train cwta and cwtp only
trainset = EEGdataset2(train_data_cwta, train_data_cwtp, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset2(test_data_cwta, test_data_cwtp, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)


model11 = copy.deepcopy(model_cwta)
model22 = copy.deepcopy(model_cwtp)
model_cwta_cwtp = MyNN_cb2(model11,model22)
model_cwta_cwtp.eval()

for name, param in model_cwta_cwtp.named_parameters():
    print(name)
    print(param)
    if name=='classifier.0.weight':
        param.data = nn.parameter.Parameter(torch.ones_like(param))
    if name=='classifier.0.bias':
        param.data = nn.parameter.Parameter(torch.zeros_like(param))

optimizer = torch.optim.SGD(model_cwta_cwtp.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop2(train_loader, model_cwta_cwtp, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop2(test_loader, model_cwta_cwtp, loss_fn)
    accs.append(tmp)
print("Done!")
accs_cwta_cwtp = accs

#%% train all
trainset = EEGdataset3(train_data_erp, train_data_cwta, train_data_cwtp, train_label)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testset = EEGdataset3(test_data_erp, test_data_cwta, test_data_cwtp, test_label)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle=True)


model11 = copy.deepcopy(model_erp)
model22 = copy.deepcopy(model_cwta)
model33 = copy.deepcopy(model_cwtp)
model_erp_cwta_cwtp = MyNN_cb3(model11,model22,model33)
model_erp_cwta_cwtp.eval()

for name, param in model_erp_cwta_cwtp.named_parameters():
    print(name)
    print(param)
    if name=='classifier.0.weight':
        param.data = nn.parameter.Parameter(torch.ones_like(param))
    if name=='classifier.0.bias':
        param.data = nn.parameter.Parameter(torch.zeros_like(param))

optimizer = torch.optim.SGD(model_erp_cwta_cwtp.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
accs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop3(train_loader, model_erp_cwta_cwtp, loss_fn, optimizer)
    #scheduler.step()
    tmp = test_loop3(test_loader, model_erp_cwta_cwtp, loss_fn)
    accs.append(tmp)
print("Done!")
accs_erp_cwta_cwtp = accs
    
plt.plot(accs_erp)
plt.plot(accs_cwta)
plt.plot(accs_cwtp)
plt.plot(accs_erp_cwta)
plt.plot(accs_erp_cwtp)
plt.plot(accs_cwta_cwtp)
plt.plot(accs_erp_cwta_cwtp)


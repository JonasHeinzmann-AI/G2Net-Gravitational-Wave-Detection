#!/usr/bin/env python
# coding: utf-8

# 
# 

# # 1.Import Library

# In[1]:


import os
import json
import random
import collections

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch

#Library for signal processing
from nnAudio.Spectrogram import CQT1992v2

import time

from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
from torch.autograd import Variable
import efficientnet_pytorch
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('logs')


# In[2]:

torch.multiprocessing.freeze_support()
print('loop')


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)


# # 2.Import Data

# In[3]:


def convert_image_id_2_path(image_id: str, is_train: bool = True) -> str:
    folder = "train" if is_train else "test"
    return "./g2net-gravitational-wave-detection/{}/{}/{}/{}/{}.npy".format(
        folder, image_id[0], image_id[1], image_id[2], image_id 
    )


# In[4]:


train_df = pd.read_csv("./g2net-gravitational-wave-detection/training_labels.csv")


# In[5]:


global_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, 
                                     hop_length=16,bins_per_octave=16,pad_mode='constant')


# # 3.Data Retriever and Data Loader
# 
# This allows us to easily get a batch of data each time we need in concatenated form.
# A batch is a set of data items. For example, each wave item has size (x,y,z). With the use of data retriever and data loader, we can easily get a batch of data which has size (b,x,y,z) with b is the batch size.

# In[6]:


class DataRetriever(torch_data.Dataset):
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets
        
        
        self.q_transform = global_transform
    def __len__(self):
        return len(self.paths)
    
    def __get_qtransform(self, x):
        image = []
        for i in range(3):
            waves = x[i] / np.max(x[i])
            waves = torch.from_numpy(waves).float()
            channel = self.q_transform(waves).squeeze().numpy()
            image.append(channel)
        out = torch.tensor(image).float()
       
        return out
    
    def __getitem__(self, index):
        file_path = convert_image_id_2_path(self.paths[index])
        x = np.load(file_path)
        image = self.__get_qtransform(x)
        
        y = torch.tensor(self.targets[index], dtype=torch.float)
            
        return {"X": image, "y": y}


# In[7]:


#Split train data frame into train set and validation set.

df_train, df_valid = sk_model_selection.train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df["target"],
)


# In[8]:


#Construct training data retriever and validation data retriever

train_data_retriever = DataRetriever(
    df_train["id"].values, 
    df_train["target"].values, 
)

valid_data_retriever = DataRetriever(
    df_valid["id"].values, 
    df_valid["target"].values,
)


# In[9]:


#This is the data loader that allow us to load batch of data

train_loader = torch_data.DataLoader(
    train_data_retriever,
    batch_size=32,
    shuffle=True,
    num_workers=12,
)

valid_loader = torch_data.DataLoader(
    valid_data_retriever, 
    batch_size=32,
    shuffle=False,
    num_workers=12,
)


# # 4.Model

# In[10]:


#This is simply pass data that Q-transformed with the size (x,y,3) to an EfficientNet, 
#the fully connected layer near the end of EfficientNet replaced by another fully connected 
#layer that serve our purpose.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        n_features = self.net._fc.in_features
        print(n_features)
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    
    def forward(self, x):
        out = self.net(x)
        return out


# # 5.Accuracy Meter and Loss Meter
# 
# This calculate the accuracy of the model and loss after each step.
# 

# In[11]:


class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg

        
class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg


# # 6.Train the model

# In[12]:

class Trainer:

    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        loss_meter, 
        score_meter
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        
        self.best_valid_score = -np.inf
        self.n_patience = 0
        
        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs."
        }
    
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            #Train the model.
            train_loss, train_score, train_time = self.train_epoch(train_loader)
            #Test the performance of trained model on validation set.
            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)
            
            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
            )
            
            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
            )

            
            self.info_message(
                self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
            )
            self.best_valid_score = valid_score
            self.save_model(n_epoch, save_path)
            self.n_patience = 0

    # MAIN FUNCTION: Traing the model       
    def train_epoch(self, train_loader):
        self.model.train() #MUST DO: set the model in Training mode because the dropout layer
        # behaves different in training and when using model to calculated result
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()
        running_loss = 0
        
        for step, batch in enumerate(train_loader, 1):
            #Load input and label
            
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            
            #Set to zero_grad to reset gradient of current parameters inside model to zero.
            #View this link for more info: https://stackoverflow.com/a/48009142
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets) #Calculate loss function, current criterion is in this link
            #https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss

            running_loss += loss.item()
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            step * len(train_loader) + step)
            
            loss.backward() #This step will calculate the gradient of model paramters and 
            #also have modification to reduce the loss

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step() #In some optimizer like Adam, learning rate and other optimize params
            #may not be constants. Therefore, they change after each step. 
            #This simple process these changes.
            
            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")
        
        return train_loss.avg, train_score.avg, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval() #MUST DO: set the model in evaluation mode when calculating output of new input.
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad(): #Because we do not need to change model parameters, we also 
                # do not need to calculate the gradient. This simple tells Pytorch that we dont want to 
                # calculate gradien and this helps reduce computational resources.
                
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)
                
            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")
        
        return valid_loss.avg, valid_score.avg, int(time.time() - t)
    
    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


# <h2> Apply above Trainer

# In[ ]:


from torchinfo import summary

def main():
    device = torch.device("cuda")
    torch.cuda.synchronize

    model = Model()

    summary(model)

    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.01)
    criterion = torch_functional.binary_cross_entropy_with_logits



    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion,
        LossMeter,
        AccMeter
    )

    history = trainer.fit(
        3,
        train_loader,
        valid_loader,
        "best-model.pth",
        100,
    )

if __name__ == '__main__':
    main()

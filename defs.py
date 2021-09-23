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



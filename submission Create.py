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
from nnAudio.Spectrogram import CQT1992v2

import time

from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
from torch.autograd import Variable
import efficientnet_pytorch

def convert_image_id_2_path(image_id: str, is_train: bool = True) -> str:
    folder = "train" if is_train else "test"
    return "./g2net-gravitational-wave-detection/{}/{}/{}/{}/{}.npy".format(
        folder, image_id[0], image_id[1], image_id[2], image_id
    )

global_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024,
                                     hop_length=16,bins_per_octave=16,pad_mode='constant')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.net(x)
        return out
device = torch.device("cpu")
model = Model()



checkpoint = torch.load("best-model.pth", map_location='cpu')
#checkpoint = torch.load("best-model.pth")
model.load_state_dict(checkpoint["model_state_dict"])



model.eval()


model = model.to(device)


class DataRetriever(torch_data.Dataset):
    def __init__(self, paths):
        self.paths = paths

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

        return torch.tensor(image).float()

    def __getitem__(self, index):
        file_path = convert_image_id_2_path(self.paths[index], is_train=False)
        x = np.load(file_path)
        image = self.__get_qtransform(x)

        return {"X": image, "id": self.paths[index]}

submission = pd.read_csv("./g2net-gravitational-wave-detection/sample_submission.csv")
submission.to_csv("submission.csv", index=False)

test_data_retriever = DataRetriever(
    submission["id"].values,
)

test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=256,
    shuffle=False,
    num_workers=12,
)
#print("1")
#model = torch.nn.DataParallel(model)
#model = model.to("cuda")
#model.cuda()
def main():

    y_pred = []
    ids = []

    for e, batch in enumerate(test_loader):
        print(f"{e}/{len(test_loader)}", end="\r")
        with torch.no_grad():
            y_pred.extend(torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze())
            ids.extend(batch["id"])

    submission = pd.DataFrame({"id": ids, "target": y_pred})
    submission.to_csv("model_submission.csv", index=False)


if __name__ == '__main__':
    main()

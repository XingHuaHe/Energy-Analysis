# Sys packages.
import os
import sys

# External package.
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class EnergyDataset(Dataset):
    def __init__(self, data_dir, samples=None, transform=None):
        if samples is None:
            self.samples = self.get_samples_data(data_dir)
        else:
            if not isinstance(samples[0], tuple):
                raise Exception("samples is not tuple")
            self.samples = samples
        
        self.transform = transform

    def __getitem__(self, index):
        sample, label = self.samples[index]
        sample = torch.from_numpy(sample).float()
        label = torch.from_numpy(label).float()
        # sample = transforms.ToTensor(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_samples_data(data_dir):
        samples = list()

        # load datas.
        datas = scio.loadmat(data_dir)
        train_dataset = datas['date']
        # select feature point.
        feature1 = train_dataset[:, 0]
        feature2 = train_dataset[:, 1]
        feature3 = train_dataset[:, 8]
        features = np.vstack((feature1, feature2, feature3)).transpose()

        # load labels.
        standard_content = datas['result']
        temp = np.zeros((standard_content.shape[0],))
        for i in range(temp.shape[0]):
            temp[i] = standard_content[i][23]
        contents = temp

        # contruct dataset.
        if features.shape[0] != contents.shape[0]:
            raise Exception("Datas count are not equal datas labels")
        for i in range(len(features)):
            samples.append((features[i], np.array(contents[i])))

        return samples
'''
# 
# Description:
#       深度回归预测模型（Pytorch）
# Author:
#       Xinghua.He
# History:
#       2020.12.27
# 
'''
# Directory package.
from utils.dataset.EnergyDataset import *
from utils.models.RegressionModel import *
from utils.processing import *

# Sys packages.
import os
import sys
import argparse
import time
import datetime

# External packages.
import tqdm
import scipy.io as scio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision.models as models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--data_path", type=str, default=None, help="data path of dataset")
    parser.add_argument("--pretraining", type=bool, default=False, help="pretraining model path")
    parser.add_argument("--pretraining_path", type=str, default="./config/resnet50-19c8e357.pth", help="pretraining model path")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    opt = parser.parse_args()
    print(opt)

    # Detetced device type (cpu or cudn).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make directory.
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    if opt.data_path is not None:
        # Setting data path according args.
        DATAPATH = opt.data_path
    else:
        # Current directory path.
        ROOTPATH = os.path.dirname(os.path.abspath(__file__))
        # Data path.
        DATAPATH = os.path.join(ROOTPATH, "..", "DatDatas", "features.mat")

    # Obtain features.
    datas = scio.loadmat(DATAPATH)
    samples = datas['features']
    features = samples[:, 0:10]
    # Obtain content.
    standard_content = samples[:, 10:13]

    # Standar features.
    xScale = preprocessing.StandardScaler()
    features = xScale.fit_transform(features, standard_content)
    # Standar contents.
    yScale = preprocessing.StandardScaler()
    standard_content = yScale.fit_transform(standard_content)

    # Split dataset.
    x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(features, standard_content, test_size=0.2, random_state=3, shuffle=True)

    train_dataset = list()
    test_dataset = list()
    for i in range(len(x_train_dataset)):
        train_dataset.append((np.array(x_train_dataset[i]), np.array(y_train_dataset[i])))
    for i in range(len(x_test_dataset)):
        test_dataset.append((np.array(x_test_dataset[i]), np.array(y_test_dataset[i])))

    # Constructed dataset.
    train_dataset = EnergyDataset(data_dir=DATAPATH, samples=train_dataset, transform=None)
    test_dataset = EnergyDataset(data_dir=DATAPATH, samples=test_dataset, transform=None)

    # Constructed dataloader.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False)
    valid_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

    # Defined model.
    model = RegressionModel(in_features=10, num_features=20).to(device)
    model.initialize_weights()

    # Defined optimizer.
    optimizer = optim.SGD(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Main training.
    for epoch in tqdm.tqdm(range(opt.epochs), ncols=100):
        model.train()
        optimizer.zero_grad()

        # Training.
        loss_mean = 0.
        for i, datas in enumerate(train_dataloader):
            samples, content = datas
            # to device.
            samples = Variable(samples.to(device))
            content = Variable(content.to(device), requires_grad=False)
            # Forward.
            outputs1, outputs2, outputs3, loss = model(samples, content)
            # Backward.
            loss.backward()
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate.
        if (epoch + 1) % 50  == 0:
            # Setting evaluated model.
            model.eval()
            target = np.zeros_like(y_test_dataset)
            predicted1 = list()
            predicted2 = list()
            predicted3 = list()
            Cr = list()
            Mn = list()
            Cu = list()
            for i, data in enumerate(tqdm.tqdm(valid_dataloader, ncols=100)):
                sample, content = data
                sample = Variable(sample.to(device), requires_grad=False)
                # content = Variable(content.to(device), requires_grad=False)
                # forward.
                pre1, pre2, pre3 = model(sample)

                predicted1.append(yScale.inverse_transform(pre1.data.to('cpu').numpy())) # <class 'list'>
                predicted2.append(yScale.inverse_transform(pre2.data.to('cpu').numpy())) # <class 'list'>
                predicted3.append(yScale.inverse_transform(pre3.data.to('cpu').numpy())) # <class 'list'>
                target[i] = yScale.inverse_transform(content.data.numpy()) # <class 'numpy.ndarray'>
            # Eventral output.
            for i in range(len(y_test_dataset)):
                Cr.append(predicted1[i][0][0] * 0.3 + predicted2[i][0][0] * 0.3 + predicted3[i][0][0] * 0.4)
            for i in range(len(y_test_dataset)):
                Mn.append(predicted1[i][0][1] * 0.3 + predicted2[i][0][1] * 0.3 + predicted3[i][0][1] * 0.4)
            for i in range(len(y_test_dataset)):
                Cu.append(predicted1[i][0][2] * 0.3 + predicted2[i][0][2] * 0.3 + predicted3[i][0][2] * 0.4)

            # Plot figures.
            _, ax = plt.subplots()
            ax.plot([i for i in range(len(y_test_dataset))], Cr)
            ax.plot([i for i in range(len(y_test_dataset))], Mn)
            ax.plot([i for i in range(len(y_test_dataset))], Cu)
            ax.plot([i for i in range(len(y_test_dataset))], target[:,0], "--")
            ax.plot([i for i in range(len(y_test_dataset))], target[:,1], "--")
            ax.plot([i for i in range(len(y_test_dataset))], target[:,2], "--")
            plt.savefig(f"./output/{epoch+1}.jpg")
            # Save data.

            # Release storage.
            del predicted1
            del predicted2
            del predicted3
            del target
            del Cr
            del Cu
            del Mn
            model.train()
    

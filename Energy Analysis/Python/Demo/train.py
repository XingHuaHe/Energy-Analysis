# System packages.
import os
import sys
import argparse
# Externed packages.
import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
# User packages.

class Energy(nn.Module):
    def __init__(self, num_class):
        super(Energy, self).__init__()
        self.num_class = num_class
        # layer 1.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((5, 5), stride=5)
        # layer 2.
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((5, 5), stride=5)
        # layer 3.
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # layer 4.
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, self.num_class)

    def forward(self, x):
        # layer 1.
        x = self.conv1(x) # 64*2046
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x) # 64*409
        # layer 2.
        x = self.conv2(x) # 128*204
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x) #128*40
        # layer 3.
        x = self.conv3(x) # 256*20
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.avgpool(x) # 256*2
        # layer 4.
        # x = self.flatten(x)
        x = torch.flatten(x, 1)
        output = self.fc1(x)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

class SpectrumImgDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.samples = self.get_samples_datas(self.data_path)
        self.transforms = transforms

    def __getitem__(self, index):
        path_img, label = self.samples[index]
        img = Image.open(path_img).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def get_samples_datas(self, data_path):
        classDirs = os.listdir(data_path)
        classPath = [os.path.join(data_path, i) for i in classDirs]

        samples = list()
        for i in range(len(classPath)):
            if classPath[i].split('\\')[-1] == "普通合金":
                img_names = os.listdir(classPath[i])
                for name in img_names:
                    path_img = os.path.join(classPath[i], name)
                    # 0 represented 普通合金.
                    samples.append((path_img, 0))
            elif classPath[i].split('\\')[-1] == "轻合金":
                img_names = os.listdir(classPath[i])
                for name in img_names:
                    path_img = os.path.join(classPath[i], name)
                    # 1 represented 轻合金.
                    samples.append((path_img, 1))
            elif classPath[i].split('\\')[-1] == "土壤":
                img_names = os.listdir(classPath[i])
                for name in img_names:
                    path_img = os.path.join(classPath[i], name)
                    # 1 represented 土壤.
                    samples.append((path_img, 2))
        return samples

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for each epoch")
    parser.add_argument("--epochs", type=int, default=100, help="epchs number")
    parser.add_argument("--datas_path", type=str, default="./outputs/")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--checkpoint_interval', type=int, default=20, help="checkpoint save interval")
    opt = parser.parse_args()

    # Detect device type.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)

    # checkpoint directry
    os.makedirs("checkpoints", exist_ok=True)

    # model.
    model = Energy(num_class=3)
    model.to(device)

    # transforms.
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # define dataset.
    train_dataset = SpectrumImgDataset(data_path=opt.datas_path+"train", transforms=train_transforms)
    test_dataset = SpectrumImgDataset(data_path=opt.datas_path+"test", transforms=test_transforms)

    # define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # define losser.
    losser = nn.CrossEntropyLoss(reduction='mean')

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # tensorboard summary
    writer = SummaryWriter(comment="2DimSpectrum")

    # train.
    for epoch in range(opt.epochs):
        model.train()
        optimizer.zero_grad()

        train_loss_mean = 0.
        train_correct = 0.
        train_total = 0.
        for _, datas in enumerate(tqdm.tqdm(train_dataloader, nrows=200)):
            imgs, labels = datas
            imgs = Variable(imgs.to(device))
            labels = Variable(labels.to(device), requires_grad=False)
            # forward
            outputs = model(imgs)
            # backward.
            loss = losser(outputs, labels)
            loss.backward()
            # update weight.
            optimizer.step()
            optimizer.zero_grad()
            # statictic.
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).squeeze().sum()
            train_loss_mean += loss.item()
            break
        # evaluate.
        model.eval()

        test_loss_mean = 0.
        test_correct = 0.
        test_total = 0.
        for _, datas in enumerate(tqdm.tqdm(test_dataloader, ncols=200)):
            imgs, labels = datas
            imgs = Variable(imgs.to(device))
            labels = Variable(labels.to(device), requires_grad=False)
            # forward
            outputs = model(imgs)
            # loss
            loss = losser(outputs, labels)
            # statictic.
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).squeeze().sum()
            test_loss_mean += loss.item()
        
        # summary.
        writer.add_scalars("Accuracy Curve" ,{
            "train" : train_correct / train_total,
            "test" : test_correct / test_total,
        }, global_step=epoch)
        writer.add_scalars("Losses Curve", {
            "train" : train_loss_mean / len(train_dataloader),
            'test' : test_loss_mean / len(test_dataloader),
        })
        # Update learning rate.
        scheduler.step()

        if (epoch+1) % opt.checkpoint_interval == 0:
            checkpoints = {
                'model' : model,
                "model_stact_dict" : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs" : epoch,
            }
            path_checkpoint = "./checkpoints/checkpoint_{}_epoch.pkl".format(epoch+1)
            torch.save(checkpoints, path_checkpoint)

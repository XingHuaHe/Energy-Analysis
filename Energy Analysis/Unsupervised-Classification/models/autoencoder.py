"""
    自编码器（2D 卷积 + 全连接）
"""

import torch
import torch.nn as nn


class AutoEncoder_512(nn.Module):
    """
        模型的输入为 512*512
    """

    def __init__(self, image_size: int, batch_size: int, trainable: bool = True) -> None:
        super().__init__()

        # attribute
        self.image_size = image_size
        self.batch_size = batch_size
        self.trainable = trainable

        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.leaky1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.leaky2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.leaky3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten  = nn.Flatten()

        # decoder
        self.liner1 = nn.Linear(in_features=128, out_features=262144)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.leaky5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.leaky6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x) -> None:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky1(x)
        x = self.conv2(x)  # 56*56
        x = self.bn2(x)
        x = self.leaky2(x)
        x = self.conv3(x)  # 18*18
        x = self.bn3(x)
        x = self.leaky3(x)  # 13*13
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        if self.trainable:
            x = self.liner1(x)
            x = torch.reshape(x, (self.batch_size, 1, 512, 512))
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.leaky5(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.leaky6(x)
            x = self.conv7(x)

        return x


class AutoEncoder_32(nn.Module):
    """
        模型的输入为 32*32
    """

    def __init__(self,image_size: int = 32, batch_size: int = 1, trainable: bool = True) -> None:
        super().__init__()
        
        # attribute
        self.image_size = image_size
        self.batch_size = batch_size
        self.trainable = trainable

        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.leaky2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(10)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten  = nn.Flatten()
        self.sigmoid1 = nn.Sigmoid()

        # decoder
        self.liner1 = nn.Linear(in_features=10, out_features=1024)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.leaky4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1)
        self.sigmoid2 = nn.Sigmoid()


    def forward(self, x) -> torch.Tensor:
        # encoder
        x = self.conv1(x)  # 32*15*15
        x = self.bn1(x)
        x = self.leaky1(x)
        x = self.conv2(x)  # 64*13*13
        x = self.bn2(x)
        x = self.leaky2(x)
        x = self.conv3(x)  # 10*6*6
        x = self.bn3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.sigmoid1(x)

        # decoder
        if self.trainable:
            x = self.liner1(x)
            x = torch.reshape(x, (self.batch_size, 1, self.image_size, self.image_size))
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.leaky4(x)
            x = self.conv5(x)
            x = self.sigmoid2(x)

        return x


if __name__ == "__main__":
    x = torch.randn((4, 3, 32, 32))
    model = AutoEncoder_32(32, 4)
    out = model(x)
    print(out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self, in_features, num_features=10):
        super(RegressionModel, self).__init__()
        # Defined feature number.
        self.num_features = num_features
        # Defined layer.
        self.linear1 = nn.Linear(in_features=in_features, out_features=num_features)
        self.linear2 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear3 = nn.Linear(in_features=num_features, out_features=3)
        self.linear4 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear5 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear6 = nn.Linear(in_features=num_features, out_features=3)
        self.linear7 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear8 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear9 = nn.Linear(in_features=num_features, out_features=3)
        # Defined loss function.
        self.MSEloss = nn.MSELoss()

    def forward(self, x, target=None):
        # layer1.
        x1 = F.leaky_relu(self.linear1(x))
        x2 = F.leaky_relu(self.linear2(x1))
        output1 = self.linear3(x2)
        # layer2.
        x4 = F.leaky_relu(self.linear4(x2))
        x = torch.add(x2, x4)
        x5 = F.leaky_relu(self.linear5(x))
        output2 = self.linear6(x5)
        # layer3.
        x7 = F.leaky_relu(self.linear7(x5))
        x = torch.add(x5, x7)
        x8 = F.leaky_relu(self.linear8(x))
        output3 = self.linear9(x8)

        if target is not None:
            loss1 = self.MSEloss(output1, target)
            loss2 = self.MSEloss(output2, target)
            loss3 = self.MSEloss(output3, target)
            loss = loss1 + loss2 + loss3

            return (output1, output2, output3, loss)
        else:
            return (output1, output2, output3)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data, 0, 0.1)
                # m.bias.data.zero_()
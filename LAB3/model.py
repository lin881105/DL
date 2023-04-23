import torch
import torch.nn as nn
import torch.nn.functional as F


class EGGNet(nn.Module):
    def __init__(self,activation = "elu"):
        super(EGGNet, self).__init__()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = (1,51), stride=(1,1), padding=(0, 25), bias = False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x.flatten(start_dim=1))
        return x

class DeepConvNet(nn.Module):
    def __init__(self,activation = "elu"):
        super(DeepConvNet, self).__init__()
        if activation == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,25,kernel_size=(1,5),stride=(1,1),padding=(0,0),bias=False),
            nn.Conv2d(25,25,kernel_size=(2,1),stride=(1,1),padding=(0,0),bias=False),
            nn.BatchNorm2d(25,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(25,50,kernel_size=(1,5),stride=(1,1),padding=(0,0),bias=False),
            nn.BatchNorm2d(50,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,5),stride=(1,1),padding=(0,0),bias=False),
            nn.BatchNorm2d(100,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100,200,kernel_size=(1,5),stride=(1,1),padding=(0,0),bias=False),
            nn.BatchNorm2d(200,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(0.5)
        )

        self.classifiy = nn.Sequential(
            nn.Linear(in_features=8600,out_features=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifiy(x.flatten(start_dim=1))

        return x

    

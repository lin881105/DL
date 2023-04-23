import dataloader
import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--eeg", help='',action="store_true")
parser.add_argument("--dconv", help='',action="store_true")
parser.add_argument("--elu", help='',action="store_true")
parser.add_argument("--leaky", help='',action="store_true")
parser.add_argument("--relu", help='',action="store_true")
parser.add_argument("--run_all", help="",action="store_true")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args =parser.parse_args()


def test(Model,activation):
    if Model == "eeg":
        net = model.EGGNet(activation).to(device)
    elif Model == "DeepConv":
        net = model.DeepConvNet(activation).to(device)

    net.load_state_dict(torch.load(f"{Model}-{activation}.pt"))
    x_train,y_train,x_test,y_test = dataloader.read_bci_data()

    test_data = dataloader.BCIDataset(x_test,y_test,device)
    test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_dataloader):

            input, label = data

            output = net(input)

            test_loss += F.cross_entropy(output,label).item()

            _,pred = torch.max(output.data, 1)

            correct += torch.sum(pred == label)
            total += label.size(0)

    print(f"Testing loss: {test_loss}")
    print(f"Testing accuracy: {correct/total}")


if __name__=="__main__":

    if args.elu:
        activation = "elu"
    elif args.relu:
        activation = "relu"
    elif args.leaky:
        activation = "leaky"

    if args.eeg:
        Model = "eeg"
    elif args.dconv:
        Model = "DeepConv"

    test(Model,activation)

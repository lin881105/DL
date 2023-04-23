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

total_epoch = 300


def train(train_dataloader, net ,epoch):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.7)

    net.train()
    print("\nEpoch", epoch)

    running_loss = 0.0
    total_correct = 0
    total = 0
    for i,data in enumerate(train_dataloader):
        input,label = data

        input.to(device)
        label.to(device)

        optimizer.zero_grad()

        output = net(input)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predict  = torch.max(output.data, 1)

        total_correct += torch.sum(predict == label)
        total += label.size(0)

    accuracy = total_correct/total

    print(f"Training loss: {running_loss}")
    print(f"Training accuracy: {accuracy}")

    return accuracy


def evaluate(test_dataloader,net):

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

    return correct/total

def run(Model, activation):

    if Model == "eeg":
        net = model.EGGNet(activation).to(device)
    elif Model == "DeepConv":
        net = model.DeepConvNet(activation).to(device)
    
    train_accuracy = []
    test_accuracy = []

    x_train,y_train,x_test,y_test = dataloader.read_bci_data()

    train_data = dataloader.BCIDataset(x_train,y_train,device)
    test_data = dataloader.BCIDataset(x_test,y_test,device)

    train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

    max = 0.0

    for epoch in range(total_epoch):
        train_accuracy.append(train(train_dataloader,net,epoch))

        test_acc = evaluate(test_dataloader,net)

        test_accuracy.append(test_acc)

        if test_acc > max:
            torch.save(net.state_dict(), f"{Model}-{activation}.pt")
            max = test_acc

    return train_accuracy,test_accuracy



if __name__=="__main__":

    if args.run_all:
        if args.eeg:
            Model = "eeg"
        elif args.dconv:
            Model = "DeepConv"
        train_acc_elu,test_acc_elu = run(Model,"elu")
        train_acc_relu,test_acc_relu = run(Model,"relu")
        train_acc_leaky,test_acc_leaky = run(Model,"leaky")
        epoch = [i for i in range(total_epoch)]


        plt.title(f"Activation function comparison({Model})")
        plt.plot(epoch, train_acc_elu, label="elu_train")
        plt.plot(epoch, test_acc_elu, label="elu_test")
        plt.plot(epoch, train_acc_relu, label="relu_train")
        plt.plot(epoch, test_acc_relu, label="relu_test")
        plt.plot(epoch, train_acc_leaky, label="leaky_relu_train")
        plt.plot(epoch, test_acc_leaky, label="leaky_relu_test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()

        print("highest test accuracy:")
        print(f"elu: {max(test_acc_elu)}")
        print(f"relu: {max(test_acc_relu)}")
        print(f"leakyrelu: {max(test_acc_leaky)}")

        
    else:
    
        x_train,y_train,x_test,y_test = dataloader.read_bci_data()
        train_data = dataloader.BCIDataset(x_train,y_train,device)
        test_data = dataloader.BCIDataset(x_test,y_test,device)

        train_dataloader = DataLoader(train_data,batch_size=128,shuffle=True)
        test_dataloader = DataLoader(test_data,batch_size=128,shuffle=True)

        if args.elu:
            activation = "elu"
        elif args.relu:
            activation = "relu"
        elif args.leaky:
            activation = "leaky"

        if args.eeg:
            Model = "eeg"
            net = model.EGGNet(activation)
        elif args.dconv:
            Model = "DeepConv"
            net = model.DeepConvNet(activation)

        for epoch in range(total_epoch):
            train(train_dataloader,net,epoch)

            evaluate(test_dataloader,net)
        
        torch.save(net.state_dict(), f"{Model}-{activation}.pt")

    


    






    





        

        




        

import torch
import numpy as np
import torch.nn as nn
import argparse
from torch import Tensor
import torchvision.models as models
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import datetime
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='',choices=["resnet18","resnet50"])
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--epoch',type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momemtum', type=float, default=0.9)
parser.add_argument('--weightDecay', type=float, default=5e-4)
parser.add_argument('--mode', choices=["train", "test"])
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--model_pth', type=str)
parser.add_argument('--run_all', action='store_true')
args = parser.parse_args()

train_data = RetinopathyLoader(f'new_train', 'train',device)
test_data = RetinopathyLoader(f'new_test', 'test',device)
train_dataloader = DataLoader(train_data, batch_size=args.batchSize, shuffle=True,num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=args.batchSize, num_workers=16)




def train(model, num_epoch):
    
    best_acc = 0.0
    best_weight = model.state_dict()
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=args.momemtum, weight_decay=args.weightDecay)
    shceduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(num_epoch):

        print('\nEpoch: ', epoch)

        running_loss = 0.0
        total_correct = 0
        total = 0
        

        for i, data in enumerate(tqdm(train_dataloader)):
            input, label = data
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predict = torch.max(output.data, 1)

            total_correct += torch.sum(predict==label)
            total += label.size(0)

        train_acc = total_correct/total

        train_acc_list.append(train_acc.cpu())
        train_loss_list.append(running_loss/total)

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(tqdm(test_dataloader)):
                input, label = data
                input = input.to(device)
                label = label.to(device)

                output = model(input)

                test_loss += criterion(output, label).item()

                _,test_pred = torch.max(output.data, 1)

                test_correct += torch.sum(test_pred==label)

                test_total += label.size(0)

        test_acc =test_correct/test_total
        test_acc_list.append(test_acc.cpu())

        test_loss_list.append(test_loss/test_total)

        if test_acc > best_acc:
            best_weight = model.state_dict()
            best_acc = test_acc

        print(f'Train Loss: {running_loss/total}')
        print(f'Training Accuracy: {train_acc}')
        print(f'Testing Loss: {test_loss/test_total}')
        print(f'Testing Accuracy: {test_acc}')
        shceduler.step()
    
    if args.pretrain:
        file_name = 'pretrain'
    else:
        file_name = 'scratch'

    torch.save(best_weight,f'checkpt/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-{args.model}-{file_name}.pt')
    
    log = {
        "train_accuracy": train_acc_list, 
        "test_accuracy": test_acc_list,
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
    }

    sio.savemat(f'log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-{args.model}-{file_name}.mat',log)

        
            
            
def test(model):

    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_dataloader):
            input, label = data
            input = input.to(device)
            label = label.to(device)

            output = model(input)

            test_loss += F.cross_entropy(output, label).item()

            _, pred = torch.max(output.data, 1)

            correct += torch.sum(pred==label)

            total += label.size(0)

    print(f'Testing Loss: {test_loss}')
    print(f'Testing accuracy: {correct/total}')





if __name__ == "__main__":
    if args.model == 'resnet18':
        if args.pretrain:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18()
    
    if args.model == 'resnet50':
        if args.pretrain:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50()


    _in = model.fc.in_features
    model.fc = nn.Linear(_in, 5)

    model = model.to(device)
    model.cuda()

    if args.mode == 'train':
        train(model, args.epoch)
    else:
        model_pth = torch.load(args.model_pth)




    
    
        
    

    

import scipy.io as sio
import matplotlib.pyplot as plt

resnet18_pretrain = sio.loadmat('log/resnet18-pretrain.mat')
resnet18_scratch = sio.loadmat('log/resnet18-scratch.mat')
resnet50_pretrain = sio.loadmat('log/resnet50-pretrain.mat')
resnet50_scratch = sio.loadmat('log/resnet50-scratch.mat')

pretrain_18_train_acc = resnet18_pretrain["train_accuracy"][0]
pretrain_18_test_acc = resnet18_pretrain["test_accuracy"][0]
pretrain_18_train_loss = resnet18_pretrain["train_loss"][0]
pretrain_18_test_loss = resnet18_pretrain["test_loss"][0]
epoch = [i for i in range(len(pretrain_18_train_acc))]

scratch_18_train_acc = resnet18_scratch["train_accuracy"][0]
scratch_18_test_acc = resnet18_scratch["test_accuracy"][0]
scratch_18_train_loss = resnet18_scratch["train_loss"][0]
scratch_18_test_loss = resnet18_scratch["test_loss"][0]



pretrain_50_train_acc = resnet50_pretrain["train_accuracy"][0]
pretrain_50_test_acc = resnet50_pretrain["test_accuracy"][0]
pretrain_50_train_loss = resnet50_pretrain["train_loss"][0]
pretrain_50_test_loss = resnet50_pretrain["test_loss"][0]
epoch50 = [i for i in range(len(pretrain_50_train_acc))]
scratch_50_train_acc = resnet50_scratch["train_accuracy"][0]
scratch_50_test_acc = resnet50_scratch["test_accuracy"][0]
scratch_50_train_loss = resnet50_scratch["train_loss"][0]
scratch_50_test_loss = resnet50_scratch["test_loss"][0]


plt.title(f"accuracy")
plt.plot(epoch, pretrain_18_train_acc, label="resnet18 pretrain/train")
plt.plot(epoch, scratch_18_train_acc, label="resnet18 scratch/train")
plt.plot(epoch, pretrain_18_test_acc, label="resnet18 pretrain/test")
plt.plot(epoch, scratch_18_test_acc, label="resnet18 scratch/test")
plt.plot(epoch50, pretrain_50_train_acc, label="resnet50 pretrain/train")
plt.plot(epoch50, scratch_50_train_acc, label="resnet50 scratch/train")
plt.plot(epoch50, pretrain_50_test_acc, label="resnet50 pretrain/test")
plt.plot(epoch50, scratch_50_test_acc, label="resnet50 scratch/test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig('learning curve.png')
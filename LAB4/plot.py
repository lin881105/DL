import scipy.io as sio
import matplotlib.pyplot as plt

log = sio.loadmat('log/20230424-054847-resnet18-pretrain.mat')

train_acc = log["train_accuracy"]
test_acc = log["test_accuracy"]
train_loss = log["train_loss"]
test_loss = log["test_loss"]
epoch = [i for i in range(len(train_acc))]
print(train_acc)

plt.title(f"accuracy")
plt.plot(epoch, train_acc, label="train")
plt.plot(epoch, test_acc, label="test")
# plt.plot(epoch, train_acc_relu, label="relu_train")
# plt.plot(epoch, test_acc_relu, label="relu_test")
# plt.plot(epoch, train_acc_leaky, label="leaky_relu_train")
# plt.plot(epoch, test_acc_leaky, label="leaky_relu_test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig('test.png')
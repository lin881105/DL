import matplotlib.pyplot as plt

with open("out-3.txt","r") as fp:
    lines = fp.readlines()

mean = []
epoch = []

for line in lines:
    x = line.split()
    for split in x:
        if split == "mean":
            epoch.append(int(x[0]))
            mean.append(float(x[3]))
            break

for i in range(len(epoch)):
    print(f'epoch: {epoch[i]} mean: {mean[i]}')


plt.plot(epoch,mean)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir")
args = parser.parse_args()


with open(os.path.join(args.log_dir,"train_record.txt"),"r") as fp:
    lines = fp.readlines()

epoch = []
beta = []
tfr = []
kld = []
mse = []
psnr = []
val_iter = []


for line in lines[1:]:
    x = line.split()
    i = 0
    for split in x:
        if split == "[epoch:":
            epoch.append(int(x[i+1][:-1]))
        if split == "mse":
            mse.append(float(x[i+2]))
        if split == 'kld':
            kld.append(float(x[i+2]))
        if split == 'tfr:':
            tfr.append(float(x[i+1]))
        if split == "beta:":
            beta.append(float(x[i+1]))
        if split == "psnr":
            psnr.append(float(x[i+2]))
            val_iter.append(len(epoch)-1)
        i+=1
            
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(epoch,mse,label='mse')
ax2.plot(epoch,kld,label="kld")
ax2.plot(epoch,beta,label='beta')
ax2.plot(epoch,tfr,label='tfr')
ax2.set_ylim(0,1.1)
ax1.plot(val_iter,psnr,label='psnr',marker='*')
ax1.set_xlabel("Epoch")

ax1.legend(loc="upper right")
ax2.legend(loc="lower right")
plt.savefig("plot.png")

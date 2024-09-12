import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

df1 = pd.read_csv('./../training_histories/history_adam_optim_alexnet_batchsize16.csv')
df2 = pd.read_csv('./../training_histories/history_adam_optim_alexnet_batchsize32.csv')
df3 = pd.read_csv('./../training_histories/history_adam_optim_alexnet_batchsize64.csv')

epochs = np.arange(1, 151)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(epochs, df1.training_loss, color='r', label=r'training; $\mathcal{B}=16$')
ax[0].plot(epochs, df2.training_loss, color='g', label=r'training; $\mathcal{B}=32$')
ax[0].plot(epochs, df3.training_loss, color='b', label=r'training; $\mathcal{B}=64$')
ax[0].plot(epochs, df1.validation_loss, linestyle='dashed', color='r', label=r'validation; $\mathcal{B}=16$')
ax[0].plot(epochs, df2.validation_loss, linestyle='dashed', color='g', label=r'validation; $\mathcal{B}=32$')
ax[0].plot(epochs, df3.validation_loss, linestyle='dashed', color='b', label=r'validation; $\mathcal{B}=64$')
ax[0].set_xlabel(r'Epochs', fontsize=15)
ax[0].set_ylabel(r'Loss', fontsize=15)

ax[1].plot(epochs, df1.training_accuracy, color='r', label=r'training; $\mathcal{B}=16$')
ax[1].plot(epochs, df2.training_accuracy, color='g', label=r'training; $\mathcal{B}=32$')
ax[1].plot(epochs, df3.training_accuracy, color='b', label=r'training; $\mathcal{B}=64$')
ax[1].plot(epochs, df1.validation_accuracy, linestyle='dashed', color='r', label=r'validation; $\mathcal{B}=16$')
ax[1].plot(epochs, df2.validation_accuracy, linestyle='dashed', color='g', label=r'validation; $\mathcal{B}=32$')
ax[1].plot(epochs, df3.validation_accuracy, linestyle='dashed', color='b', label=r'validation; $\mathcal{B}=64$')
ax[1].set_xlabel(r'Epochs', fontsize=15)
ax[1].set_ylabel(r'Accuracy', fontsize=15)

ax[0].grid()
ax[1].grid()
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
plt.suptitle('AlexNet, Adam optimizer', fontsize=18)
#plt.savefig('history_adam_optim_alexnet.png', dpi=300, bbox_inches='tight')
plt.show()
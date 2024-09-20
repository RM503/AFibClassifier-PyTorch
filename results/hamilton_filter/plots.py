import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

folder_path = './../../training_histories/hamilton_filter/'

df1 = pd.read_csv(folder_path + 'history_sgd_optim_alexnet_batchsize64.csv')
df2 = pd.read_csv(folder_path + 'history_adam_optim_alexnet_batchsize64.csv')
df3 = pd.read_csv(folder_path + 'history_adamw_optim_alexnet_batchsize64.csv')

epochs = np.arange(1, 151)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(epochs, df1.training_loss, color='r', label=r'training; SGD')
ax[0].plot(epochs, df2.training_loss, color='g', label=r'training; Adam')
ax[0].plot(epochs, df3.training_loss, color='b', label=r'training; AdamW')
ax[0].plot(epochs, df1.validation_loss, linestyle='dashed', color='r', label=r'validation; SGD')
ax[0].plot(epochs, df2.validation_loss, linestyle='dashed', color='g', label=r'validation; Adam')
ax[0].plot(epochs, df3.validation_loss, linestyle='dashed', color='b', label=r'validation; AdamW')
ax[0].set_xlabel(r'Epochs', fontsize=15)
ax[0].set_ylabel(r'Loss', fontsize=15)

ax[1].plot(epochs, df1.training_accuracy, color='r', label=r'training; SGD')
ax[1].plot(epochs, df2.training_accuracy, color='g', label=r'training; Adam')
ax[1].plot(epochs, df3.training_accuracy, color='b', label=r'training; AdamW')
ax[1].plot(epochs, df1.validation_accuracy, linestyle='dashed', color='r', label=r'validation; SGD')
ax[1].plot(epochs, df2.validation_accuracy, linestyle='dashed', color='g', label=r'validation; Adam')
ax[1].plot(epochs, df3.validation_accuracy, linestyle='dashed', color='b', label=r'validation; AdamW')
ax[1].set_xlabel(r'Epochs', fontsize=15)
ax[1].set_ylabel(r'Accuracy', fontsize=15)

ax[0].grid()
ax[1].grid()
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')

#plt.savefig('history_adam_optim_alexnet.png', dpi=300, bbox_inches='tight')
plt.show()
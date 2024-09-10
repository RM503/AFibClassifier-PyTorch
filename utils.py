''' 
Dataset class and other helper functions
'''

import os
import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from torchvision.io import read_image 
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pixel_stats(annotations_file, img_dir):
    ''' 
    This function calculates the mean and standard deviation of the pixels across the R, G and B color channels.
    It returns the mean and standard deviation as torch.tensor() objects.
    '''
    img_labels = annotations_file.iloc[:,0] # dataframe converted into a series object here

    channel_sum = 0
    channel_squared_sum = 0

    for idx in range(len(img_labels)):
        img_file_name = img_labels[idx] + '.png'
        img_path = os.path.join(img_dir, img_file_name)
        img = read_image(img_path)[0:3, :, :].float()

        ''' 
        When the mean and standard deviation of the channels are calculated, the image matrix must be
        converted to the appropriate type using .float(). Otherwise, PyTorch will return an error message.
        '''
        channel_sum += torch.mean(img.float(), dim=[1, 2]) # dim=1,2 refer to the pixel axes only; returns a 1 x 3 tensor
        channel_squared_sum += torch.mean((img**2).float(), dim=[1,2])

    mean = channel_sum/len(img_labels)
    std = torch.sqrt(( channel_squared_sum/len(img_labels) - mean**2 ))

    return mean, std 

class ScalogramDataset(Dataset):
    ''' 
    The scalogram dataset is initialized with the REFERENCE.csv file, image directory and a list of 
    suitable transformations on the image dataset.

    In this form, the annotations_file is represented as a directory reference to the REFERENCE.csv file. The split index argument refers to
    the particular train/test split inside the image directory.
    '''
    def __init__(self, annotations_file, img_dir, split_index, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=[0]).iloc[split_index, :]
        self.img_dir = img_dir 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)[0:3, :, :].float()
        label = self.img_labels.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label 

@torch.no_grad()
def get_confusion_matrix(validation_dataset, validation_loader, model, plot=False):
    '''
    Inputs - the validation/test dataset, dataloader and the instantiated NN model

    Returns the confusion matrix of the predicted and true class labels
    '''

    all_preds = torch.Tensor([])

    for image, label in validation_loader:
        image, label = image.to(device), label.to(device)
        output = model(image)

        all_preds = torch.cat((all_preds, output.data), dim=0) # vertically stacks all outputs from iterations

    '''
    Converting the PyTorch tensors into numpy arrays
    '''
    all_preds = torch.max(all_preds, dim=1).indices.numpy()
    all_labels = torch.Tensor(validation_dataset.targets).numpy()
    
    CM =  confusion_matrix(all_labels, all_preds)

    if plot is False:
        return CM 
    else:
        class_labels = [
        
    ]

    sns.heatmap(CM, annot=True, fmt='.3g', xticklabels=class_labels, yticklabels=class_labels)
    plt.show()
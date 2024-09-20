''' 
This module contains the custom PyTorch dataset class along with other helper functions.
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

training_img_dir = './signal_cwt_images_training/' # image directory
label_list = pd.read_csv(training_img_dir + 'REFERENCE.csv', index_col=[0]) # annotations file

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
        image = read_image(img_path)[0:3, :, :].float() # by default, the image tensors are uint8; these are converted to floats
        label = self.img_labels.iloc[idx, 2] # the third column contains the encoded diagnoses

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label 

@torch.no_grad()
def get_confusion_matrix(idx_test, validation_loader, model, plot=False):
    '''
    This function takes in as arguments a list/array of indicies of the test/validation data, the test/validation dataloader
    and the trained NN model and returns the confusion matrix as a 4 x 4 array. Optionally, if plot=True, the function 
    returns a heatmap plot of the confusion matrix.
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
    all_labels = label_list.iloc[idx_test, 2].to_numpy()
    
    CM =  confusion_matrix(all_labels, all_preds)

    if plot is False:
        return CM 
    else:
        class_labels = ['AFib', 'Normal', 'Other', 'Noise']

    sns.heatmap(CM, annot=True, fmt='.3g', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.xlabel(r'predicted labels', fontsize=15)
    plt.ylabel(r'true labels', fontsize=15)
    plt.show()

def F1_score(CM, class_label):
    '''
    This function returns the F1 score associated with a particular class label.
    0 - AFib, 1 - Normal, 2 - Other, 3 - Noise
    '''
    keys = {
        'AFib'   : 0,
        'Normal' : 1,
        'Other'  : 2,
        'Noise'  : 3
    }
    key = keys[class_label]
    ''' 
    The F1 score is calculate as the harmonic mean of precision and recall

    F1 = 2/(1/precision + 1/recall) = 2*precision*recall / (precision + recall)
    '''
    precision = CM[key, key] / CM[:, key].sum()
    recall = CM[key, key] / CM[key, :].sum()
    F1_score = 2*precision*recall / (precision + recall) # F1 score associated with particular key

    return F1_score 

def prediction_stats(CM):
    ''' 
    This function uses the confusion matrix and produces the precision, recall and F1 scores of each category.
    The scores are then returned as a dataframe for better readability.
    '''
    keys = {
        'AFib'   : 0,
        'Normal' : 1,
        'Other'  : 2,
        'Noise'  : 3
    }
    precision_list = []
    recall_list = []
    F1_list = []

    for key, val in keys.items():
        precision = CM[val, val] / CM[val, :].sum()
        precision_list.append(precision)
        recall = CM[val, val] / CM[:, val].sum()
        recall_list.append(recall)
        f1 = 2*precision*recall / (precision + recall)
        F1_list.append(f1)

    stats_dict = {'Precision' : precision_list, 'Recall': recall_list, 'F1_score': F1_list}
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['AFib', 'Normal', 'Other', 'Noise'])

    return stats_df 
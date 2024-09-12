''' 
This is the main program file where the model is trained. Certain input parameters are to be passed through
the command line using argparese.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from utils import pixel_stats, ScalogramDataset
from models import AlexNet # imports the AlexNet NN class 
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit_one_cycle(num_epochs, max_lr, model, train_dl, valid_dl, loss_fn, opt_fun, weight_decay=0):
    '''
    Defining one complete cycle of training and validation over the specified number of epochs
    The function takes in the following parameters
    (1) number of epochs - num_epochs
    (2) maximum learning rate - max_lr
    (3) neural network model
    (4) training and validation/test dataloaders
    (5) regularization in terms of weigh decay (default set to 0)

    '''

    # instantiate optimizer with appropriate weight_decay
    optimizer = opt_fun(model.parameters(), max_lr, weight_decay=weight_decay)
    # setting up one-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_dl)
    )
    ''' 
    Create empty lists to store training and validation histories
    '''
    train_size = len(train_dl.dataset)
    valid_size = len(valid_dl.dataset)
    loss_train = [] # Loss history per epoch during training
    acc_train  = [] # Accuracy history per epoch during training
    loss_valid = [] # Loss history per epoch during validation
    acc_valid  = [] # Accuracy history per epoch during validation

    keys = ['epoch', 'training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy', 'last_lr']
    history = {key : [] for key in keys}    # initialize empty dictionary to store training and validation information and save as csv file 

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        lrs = [] # for storing the adaptive learning rates per epoch

        for image, label in tqdm(train_dl, desc='Training loop'):
            
            '''  
            Calculate prediction and loss function per epoch
            '''
            output = model(image)
            loss = loss_fn(output, label)
            ''' 
            Backpropagating the loss function to re-adjust weights
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            Recording and updating learning rates
            '''
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()*image.size(0) # Accumulated loss per training batch
            _, predicted = torch.max(output, dim=1)
            running_total += label.size(0)
            running_correct += (predicted == label).sum().item()

        training_loss = running_loss/train_size
        loss_train.append(training_loss)
        training_accuracy = running_correct/running_total # Proportion of accurate predictions 
        acc_train.append(training_accuracy)
        '''
        Validation phase
        '''
        running_loss = 0.0
        running_correct = 0 # Keeps track of the number of correct classifications
        running_total = 0

        with torch.no_grad():
            for image, label in tqdm(valid_dl, desc='Validation loop'):
                model.eval()
                output = model(image)
                loss = loss_fn(output, label)
                
                running_loss += loss.item()*image.size(0) # Accumulated loss per validation batch
                _, predicted = torch.max(output, dim=1)
                running_total += label.size(0)
                running_correct += (predicted == label).sum().item()
        validation_loss = running_loss/valid_size 
        loss_valid.append(validation_loss)
        validation_accuracy = running_correct/running_total
        acc_valid.append(validation_accuracy)

        # Append information in csv file
        history['epoch'].append(epoch)
        history['training_accuracy'].append(training_accuracy)
        history['training_loss'].append(training_loss)
        history['validation_accuracy'].append(validation_accuracy)
        history['validation_loss'].append(validation_loss)
        history['last_lr'].append(lrs[-1])

        # Epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}: trainining loss = {training_loss:4f}, last_lr = {lrs[-1]}, training accuracy = {training_accuracy:4f},\
               validation loss = {validation_loss:4f}, validation accuracy = {validation_accuracy:4f}')
        
    history_df = pd.DataFrame.from_dict(history)
    #history_df.to_csv('./training_histories/history.csv')  # modify the training history file with specifics of the model and parameters
    
    return loss_train, acc_train, loss_valid, acc_valid

if __name__ == '__main__':
    ''' 
    The code will be run here. The following parameters are to be passed as inputs using argparse -
    (1) num_epochs (mandatory, no default value set)
    (2) batch_size (optional, default=64)
    (3) max_lr (optional, defaul=0.001)
    (4) weight_decay (optional, default=0.0001)
    '''

    parser = argparse.ArgumentParser()
    
    parser.add_argument('num_epochs', type=int, help='Number of epochs of training')
    parser.add_argument('-n1', '--batch_size', type=int, default=64,  help='Batch size for PyTorch DataLoader')
    parser.add_argument('-n2', '--max_lr', type=float, default=0.01, help='Maximum learning rate for scheduler')
    parser.add_argument('-n3', '--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_lr = args.max_lr
    weight_decay = args.weight_decay
    
    print(f'Number of epochs = {args.num_epochs}')
    print(f'Batch size = {args.batch_size}')
    print(f'Maximum learning rate = {args.max_lr}')
    print(f'Weight decay = {args.weight_decay}')

    '''
    Before the dataset is loaded, a train/test split is created on the data indices. The list of train and test indices are then used 
    to create training and validation datasets on which transformations are applied before being passed onto the dataloader.
    '''

    training_img_dir = './signal_cwt_images_training/' # image directory
    label_list = pd.read_csv(training_img_dir + 'REFERENCE.csv', index_col=[0]) # annotations file

    idx = np.arange(0, len(label_list)) # list of indices
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        shuffle=True 
    ) # test size of 20% is chosen

    # Derive pixel stats, convert them to lists and create a tuple
    mean, std = pixel_stats(label_list, training_img_dir)
    mean = mean.tolist()
    std = std.tolist()
    img_stats = (mean, std)

    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(227, 227)),
            transforms.RandomCrop(size=(227, 227), padding=4, padding_mode='reflect'),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize(*img_stats, inplace=True)
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(size=(227, 227)),
            transforms.Normalize(*img_stats, inplace=True)
        ]
    )

    train_ds = ScalogramDataset(
        training_img_dir + 'REFERENCE.csv',
        training_img_dir,
        idx_train,
        transform=train_transform
    )

    valid_ds = ScalogramDataset(
        training_img_dir + 'REFERENCE.csv',
        training_img_dir,
        idx_test,
        transform=valid_transform
    )

    print(f'Size of the train set {len(train_ds)}.')
    print(f'Size of the test dataset is {len(valid_ds)}.')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = AlexNet().to(device)

    #num_epochs = 150
    #max_lr = 0.001
    #weight_decay = 0.0001
    opt_fun = torch.optim.Adam

    loss_fn = nn.CrossEntropyLoss()

    print('Training has started.')

    loss_train, acc_train, loss_valid, acc_valid = fit_one_cycle(
        num_epochs,
        max_lr,
        model,
        train_dl,
        valid_dl,
        loss_fn,
        opt_fun,
        weight_decay=weight_decay
    )
    print('Training has finished.')

    #FILE = './saved/afibclass_adamoptim_alexnet_batchsize' + str(batch_size) + '.pth'
    #torch.save(model.state_dict(), FILE)
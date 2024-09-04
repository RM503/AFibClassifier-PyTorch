''' 
Python script for preprocessing the ECG data
'''

import numpy as np
import h5py 
import wfdb 
import os 
import glob 

def fix_signal_length(file_list):
    '''
    This function fixes the lengths of the ECG data. The maximum recording length is 60s, which at a sampling rate of 300 Hz, gives a signal length
    of 18,000. Readings that are given by arrays of length less than 18,000 are padded with zeros to the right. 

    The function takes as input a directory of all the .mat and .hea files from which the signals are extracted and flattened. Each of the padded
    array as stored in a 2d array that is returned as output.
    '''
    data_array = np.zeros((len(file_list), 18000)) # all the different signals are stored along the rows

    for i in range(len(file_list)):

        record = wfdb.rdrecord(file_list[i]).__dict__
        signal = record['p_signal'].flatten()
        sig_len = record['sig_len']
        
        if sig_len < 18000:
            extra_dim = 18000 - sig_len # extra padding dimensions
            signal_padded = np.pad(signal, (0, extra_dim), 'constant', constant_values=0)
        data_array[i, :] = signal_padded 

    return data_array 

def makeHDF5(data_array, file_list):
    ''' 
    This function takes the signal array and the file list and packs them into an HDF5 file.
    '''

    with h5py.File('./data/data.h5') as f:
        for i in range(len(file_list)):
            key = wfdb.rdrecord(file_list[i]).__dict__['record_name']
            f.create_dataset(key, data=data_array[i, :])

if __name__ == '__main__':
    file_list = sorted(glob.glob('./data/*.mat'))
    file_list = [os.path.splitext(x)[0] for x in file_list]

    #print(wfdb.rdrecord(data_list[0]).__dict__)
    
''' 
This Python script preprocesses the ECG data by
(i) fixing the signal length of each sample
(ii) encapsulating them in an HDF5 file
(iii) converting the signals to scalograms in parallelized loops
'''

import numpy as np
import h5py 
import wfdb 
import os 
import glob
import neurokit2 as nk
import pywt 
import matplotlib.pyplot as plt
from tqdm import tqdm
import time 
from joblib import Parallel, delayed

def fix_signal_length(file_list):
    '''
    This function fixes the lengths of the ECG data. The maximum recording length is slightly longer tha 60s, which at a sampling rate of 300 Hz, gives a signal length
    of over 18,200. Readings that are given by arrays of length less than 18,000 are padded with zeros to the right. 

    The function takes as input a directory of all the .mat and .hea files from which the signals are extracted and flattened. Each of the padded
    array as stored in a 2d array that is returned as output.

    There are a few entries which slightly exceed 60s (hence, have more than 18,000 data points). Such entries have been restricted to array size of
    18,000.
    '''
    data_array = np.zeros((len(file_list), 18000)) # all the different signals are stored along the rows

    for i in range(len(file_list)):

        record = wfdb.rdrecord(file_list[i]).__dict__
        signal = record['p_signal'].flatten() # signals are stored as (N,1) arrays; flatten them to 1d
        sig_len = record['sig_len']
        
        if sig_len < 18000:
            extra_dim = 18000 - sig_len # extra padding dimensions
            signal_padded = np.pad(signal, (0, extra_dim), 'constant', constant_values=0)
            data_array[i, :] = signal_padded

        elif sig_len == 18000:
            data_array[i, :] = signal # no padding if sig_len = 18000

        elif sig_len > 18000:
             data_array[i, :] = signal[0:18000] # truncate signal length to 18000

    return data_array 

def make_HDF5(data_array, file_list):
    ''' 
    This function takes the signal array and the file list and packs them into an HDF5 file.
    '''

    with h5py.File('./data/data.h5', 'w') as f:
        for i in range(len(file_list)):
            key = wfdb.rdrecord(file_list[i]).__dict__['record_name']
            f.create_dataset(key, data=data_array[i, :])

def signal_CWT(key, sampling_rate=300, method='neurokit', wavelet='cmor2.5-1.0'):
    '''
    This function imports the HDF5 data file and iterates through each signal by key. Eah signal is
    cleaned using neurokit's nk.ecg_clean() function with 'neurokit' as the default method. The cleaned
    signal is then pass through pywt continuous wavelet transform function with the complex Morlet wavelet
    used as default. The scalogram plots for each signal are saved in a separate directory as .png images
    which are represented by 224 x 224 x 3 matrices.
    '''
    with h5py.File('./data/data.h5', 'r') as f:
        raw_signal = np.array(f[key])
        sampling_times = np.linspace(0.0, 60.0, 18000)
        clean_signal = nk.ecg_clean(raw_signal, sampling_rate, method)

        ''' 
        Use f = scale2frequency(wavelet, scale)/sampling_period to check frequencies corresponding to the chosen scales
        '''
        scales = np.linspace(10, 750, num=150) # choosing wavelet frequencies between 0.5 Hz to 150 Hz
        sampling_period = np.diff(sampling_times).mean()

        # continuous wavelet transform
        coeff_mat, freqs = pywt.cwt(
            clean_signal,
            scales=scales,
            wavelet=wavelet,
            sampling_period=sampling_period
        ) # the cwt() function returns frequencies and coefficient matrices

        coeff_mat = np.abs(coeff_mat)
        fig, ax = plt.subplots(figsize=(2.90, 2.91))

        pcm = ax.pcolormesh(sampling_times, freqs, coeff_mat, cmap='jet')
        ax.set_yscale('log')
        ax.axis('off')

        plt.savefig('./signal_cwt_images_training/' + key + '.png', dpi=100, bbox_inches='tight', pad_inches=0.0) # images saved in new directory
        plt.close(fig)

if __name__ == '__main__':
    file_list = sorted(glob.glob('./data/*.mat'))
    file_list = [os.path.splitext(x)[0] for x in file_list]

    ''' 
    The signals are set to a fixed length (padding and trunctation), stored in a 2d array and converted into an
    HDF5 file for convenience.
    '''
    #data_array = fix_signal_length(file_list)
    #make_HDF5(data_array, file_list)

    ''' 
    The signal to scalogram conversions are performed in parallel using joblib.
    '''

    with h5py.File('./data/data.h5') as f:
        keys = list(f.keys())

        t1 = time.time()

        Parallel(n_jobs=8)(delayed(signal_CWT)(key) for key in tqdm(keys)) # joblib's parallelized for loop

        t2 = time.time()
        print('Time taken for execution: ' + str(t2-t1) + ' seconds')
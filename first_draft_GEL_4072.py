import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn import preprocessing


# Loading recorded EMG signals into numpy arrays
gesture_0_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-000.csv', delimiter=',')
gesture_0_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-001.csv', delimiter=',')
gesture_0_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-002.csv', delimiter=',')

gesture_1_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-000.csv', delimiter=',')
gesture_1_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-001.csv', delimiter=',')
gesture_1_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-002.csv', delimiter=',')

gesture_2_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-000.csv', delimiter=',')
gesture_2_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-001.csv', delimiter=',')
gesture_2_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-002.csv', delimiter=',')

gesture_3_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-000.csv', delimiter=',')
gesture_3_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-001.csv', delimiter=',')
gesture_3_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-002.csv', delimiter=',')

gesture_4_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-000.csv', delimiter=',')
gesture_4_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-001.csv', delimiter=',')
gesture_4_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-002.csv', delimiter=',')

gesture_5_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-000.csv', delimiter=',')
gesture_5_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-001.csv', delimiter=',')
gesture_5_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-002.csv', delimiter=',')

def getMAV(x):
    '''
    Computes the Mean Absolute Value (MAV)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Mean Absolute Value as [float]
    '''
    MAV = np.mean(np.abs(x))
    return MAV

def getRMS(x):
    '''
    Computes the Root Mean Square value (RMS)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Root Mean Square value as [float]
    '''
    RMS = np.sqrt(np.mean(x**2))
    return RMS

def getVar(x):
    '''
    Computes the Variance of EMG (Var)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Variance of EMG as [float]
    '''
    N = np.size(x)
    Var = (1/(N-1))*np.sum(x**2)
    return Var

def getSD(x):
    '''
    Computes the Standard Deviation (SD)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Standard Deviation as [float]
    '''
    N = np.size(x)
    xx = np.mean(x)
    SD = np.sqrt(1/(N-1)*np.sum((x-xx)**2))
    return SD

def getZC(x, threshold=0):
    '''
    Computes the Zero Crossing value (ZC)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Zero Crossing value as [float]
    '''
    N = np.size(x)
    ZC=0
    for i in range(N-1):
        if (x[i]*x[i+1] < 0) and (np.abs(x[i]-x[i+1]) >= threshold):
            ZC += 1
    return ZC

def getSSC(x, threshold=0):
    '''
    Computes the Slope Sign Change value (SSC)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Slope Sign Change value as [float]
    '''
    N = np.size(x)
    SSC = 0
    for i in range(1, N-1):
        if (((x[i] > x[i-1]) and (x[i] > x[i+1])) or ((x[i] < x[i-1]) and (x[i] < x[i+1]))) and ((np.abs(x[i]-x[i+1]) >= threshold) or (np.abs(x[i]-x[i-1]) >= threshold)):
            SSC += 1
    return SSC

def segment_dataset(filepath, window_length=200, cha0=6, cha1=17, classes=None):
    files = os.listdir(filepath)
    fileNames = []
    data = []
    labels = []
    for f in files:
        if f.endswith('.csv'):
            fileNames.append(f)

    for i in fileNames:
        fileName, fileType = i.split('.')
        metaData = fileName.split('-')      # [0]: Gesture/Label, [1]: Trial

        if np.in1d(int(metaData[0]), classes):
          data_read_ch0 = np.loadtxt(filepath+i, delimiter=',')[cha0]  # Choosing channel 6 as first channel for this exercise
          data_read_ch1 = np.loadtxt(filepath+i, delimiter=',')[cha1] # Choosing channel 17 as second channel for this exercise

          len_data = len(data_read_ch0)
          n_window = int(len_data/window_length)

          data_windows_ch0 = [data_read_ch0[w*window_length:w*window_length+window_length] for w in range(n_window)]
          data_windows_ch1 = [data_read_ch1[w*window_length:w*window_length+window_length] for w in range(n_window)]

          data += [(a, b) for a, b in zip(data_windows_ch0, data_windows_ch1)]

          labels += [int(metaData[0])]*n_window
        else:
          pass

    return data, labels

def features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True):
    dataset = []
    for d in data:
        feature_vector = []
        if MAV==True:
            feature_vector += [getMAV(d[0])]
            feature_vector += [getMAV(d[1])]
        if RMS==True:
            feature_vector += [getRMS(d[0])]
            feature_vector += [getRMS(d[1])]
        if Var==True:
            feature_vector += [getVar(d[0])]
            feature_vector += [getVar(d[1])]
        if SD==True:
            feature_vector += [getSD(d[0])]
            feature_vector += [getSD(d[1])]
        if ZC==True:
            feature_vector += [getZC(d[0])]
            feature_vector += [getZC(d[1])]
        if SSC==True:
            feature_vector += [getSSC(d[0])]
            feature_vector += [getSSC(d[1])]
        dataset += [feature_vector]
    return dataset

def plot_emg_and_frequency_content(signal, fs):
    '''
    This function plots an emg signal and its frequency content
    param signal : EMG signal measured from a hand gesture
    param fs : frequency sample
    return : None
    '''
    ps = np.abs(np.fft.fft(signal))**2
    time_step = 1/fs
    freqs = np.fft.fftfreq(signal.size, time_step)
    idx = np.argsort(freqs)
    print(np.max(signal), np.min(ps[idx]))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].plot(signal)
    axes[0].set_title('EMG signal')
    axes[0].set(xlabel=None, ylabel=None, xlim=[0, None], ylim=[0.9*np.min(signal), 1.1*np.max(signal)])
    axes[1].plot(freqs[idx], ps[idx])
    axes[1].set_title('FFT')
    axes[1].set(xlabel=None, ylabel=None, xlim=[0, 500], ylim=[0.9*np.min(ps[idx]), 1.1*np.max(ps[idx])])
    fig.tight_layout()

    plt.show()
    return
plot_emg_and_frequency_content(gesture_0_0[6], 1000)


path = 'all_data/data_2_electrode_GEL_4072/'

classes = [0, 1, 2, 3, 4, 5]

data, labels = segment_dataset(path, window_length=150, cha0=6, cha1=17, classes=classes)
features_set = features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True)
features_set = preprocessing.scale(features_set) # preprocessing module imported from sklearn


feat_x = 2  # 0-1: MAV, 2-3: RMS, 4-5: Var, 6-7: SD, 8-9: ZC, 10-11: SSC
feat_y = 3

for c in classes:
  ind = np.where(np.array(labels)==c)
  plt.scatter([f[feat_x] for f in features_set[ind]], [f[feat_y] for f in features_set[ind]], label='Class '+str(c))
plt.legend()
plt.show()




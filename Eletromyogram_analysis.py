import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Electromyogram_analysis:
    """
    Classe qui va traiter les électromyogrammes en fonction du type de fichier qu'il reçoit.
    Dans ce projet, on va traiter des électromyogrammes venant de deux sources différentes, qui ont sauvegardés leurs signaux de façon différente (csv ou .mat)
    Dans les deux cas, on va traiter les données et les reformat pour produire un fichier .txt, dans lequel on va sauvegarder un dictionnaire.
    Celui-ci va contenir notre jeu et nos targets dans un format que l'on est plus habitué (avec array numpy ou tenseurs pytorch)
    """
    def __init__(self, path, f_types):
        """init class pour path, f_types et sample_frequency"""
        self.path = path
        self.f_types = f_types
        self.sample_frequency = 1000
        self.data = {}
    

    def getMAV(self, x, axis=None):
        """
        Computes the Mean Absolute Value (MAV)
        :param x: EMG signal vector as [1-D numpy array]
        :return: Mean Absolute Value as [float]
        """
        MAV = np.mean(np.abs(x), axis)
        return MAV
    

    def getRMS(self, x, axis=None):
        """
        Computes the Root Mean Square value (RMS)
        :param x: EMG signal vector as [1-D numpy array]
        :return: Root Mean Square value as [float]
        """
        RMS = np.sqrt(np.mean(x**2, axis))
        return RMS
    
    
    def getVAR(self, x, axis=None):
        """
        Computes the Variance of EMG (Var)
        :param x: EMG signal vector as [1-D numpy array]
        :return: Variance of EMG as [float]
        """
        N = np.shape(x)[-1]
        Var = (1/(N-1))*np.sum(x**2, axis)
        return Var
    
    
    def getSD(self, x, axis=None):
        """
        Computes the Standard Deviation (SD)
        :param x: EMG signal vector as [1-D numpy array]
        :return: Standard Deviation as [float]
        """
        N = np.shape(x)[-1]
        xx = np.mean(x, axis)
        SD = np.sqrt(1/(N-1)*np.sum((x-xx)**2, axis))
        return SD


    def int_to_mvmnt_csv(self):
        """description"""
        assert self.f_types == 'csv'
        mvmnts = {0: 'Closed fist', 1: 'Thumb up', 2: 'Tri-pod pinch', 3: 'Neutral hand position', 4: 'Fine pinch (index+thumb)', 5: 'Pointed index'}
        return mvmnts.get(self.target)
    
    
    def format_csv_files(self, window_length=150, classes=None):
        """ format csv files and makes sure files ext are .csv """
        assert self.f_types == 'csv'
        
        list_of_csv_files = os.listdir(self.path)
        target = []

        for file in list_of_csv_files:
            """ Create target array and figure out shape of data"""
            # file names : xxx-yyy
            # xxx : Mouvement
            # yyy : trial
            data_read = np.loadtxt(self.path + '/' + file, delimiter=',')
            len_data = np.shape(data_read)[1]
            n_window = int(len_data/window_length)
            target += [int(file[0:3])]*n_window

        n_row = np.shape(target)[0]
        n_col = np.shape(data_read)[0]
        data = [[[]]*n_col]*n_row
        count = 0
        for file in list_of_csv_files:
            """ fill data. data[0][1] : emg # 0 of electrode 1, it has a size = window_length """
            data_read = np.loadtxt(self.path + '/' + file, delimiter=',')
            len_data = np.shape(data_read)[1]
            n_window = int(len_data/window_length)
            for w in range(n_window):
                for electrode in range(n_col):
                    data[w+count][electrode] = data_read[electrode][w*window_length:w*window_length+window_length]
            count += w
 
        self.data = data
        self.target = target
    
    def create_features_dataset(self):
        """description"""
        #features_data = {'ch0' : {'mav' : [], 'rms' : [], 'var' : [], 'sd' : []}, 'ch1' : {'mav' : [], 'rms' : [], 'var' : [], 'sd' : []}}
        for d in self.data:
            features_data['mav'] += [self.getMAV(d[0])]
            features_data['rms'] += [self.getRMS(d[0])]
            features_data['var'] += [self.getVAR(d[0])]
            features_data['sd'] += [self.getSD(d[0])]
        self.features_data = features_data


    
    def format_mat_files(self):
        """ format mat files and makes sure files ext are .mat """
        assert self.f_types == 'mat'
        return


    def plot_emg_signal_and_fft(self, emg_signal):
        """ Affiche une figure contenant le signal emg à gauche et sa transformée de fourier à droite """
        # to do : axes : temps, intensité, fréquences, intensité
        ps = np.abs(np.fft.fft(emg_signal))**2
        time_step = 1/self.sample_frequency
        freqs = np.fft.fftfreq(emg_signal.size, time_step)
        idx = np.argsort(freqs)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        axes[0].plot(emg_signal)
        axes[0].set_title('EMG signal')
        axes[0].set(xlabel=None, ylabel=None, xlim=[0, None], ylim=[0.9*np.min(emg_signal), 1.1*np.max(emg_signal)])
        axes[1].plot(freqs[idx], ps[idx])
        axes[1].set_title('FFT')
        axes[1].set(xlabel=None, ylabel=None, xlim=[0, 500], ylim=[0.9*np.min(ps[idx]), 1.1*np.max(ps[idx])])
        fig.tight_layout()

        plt.show()


    def plot_hitogram_mvmnts(self, dataset):
        """ Affiche un histogramme représentant la répartition des mouvements provenant du dataset"""
        return
    


path = 'all_data/data_2_electrode_GEL_4072'
f_types = 'csv'
test = Electromyogram_analysis(path, f_types)
test.format_csv_files()


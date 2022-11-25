import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams.update({'font.size': 16})

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


    def int_to_mvmnt_csv(self, mvmnt):
        """description"""
        assert self.f_types == 'csv'
        mvmnts = {0: 'Closed fist', 1: 'Thumb up', 2: 'Tri-pod pinch', 3: 'Neutral hand position', 4: 'Fine pinch (index+thumb)', 5: 'Pointed index'}
        return mvmnts.get(mvmnt)
    
    
    def format_csv_files(self, window_length=200):
        """ format csv files and makes sure files ext are .csv """
        assert self.f_types == 'csv'
        
        list_of_csv_files = os.listdir(self.path)
        n_row = 0
        for file in list_of_csv_files:
            """ Create target array and figure out shape of data"""
            # file names : xxx-yyy
            # xxx : Mouvement
            # yyy : trial
            data_read = np.loadtxt(self.path + '/' + file, delimiter=',')
            len_data = np.shape(data_read)[1]
            n_window = int(len_data/window_length)
            n_row += n_window

        n_col = np.shape(data_read)[0]

        data = [[[]]*n_col]*n_row
        target = []
        mav_data = np.zeros((n_row, n_col))
        rms_data = np.zeros((n_row, n_col))
        var_data = np.zeros((n_row, n_col))
        sd_data = np.zeros((n_row, n_col))
        count = 0
        for file in list_of_csv_files:
            """ fill self.emg_data : [0][1] : emg # 0 of electrode 1, it has a size = window_length """
            data_read = np.loadtxt(self.path + '/' + file, delimiter=',')
            len_data = np.shape(data_read)[1]
            n_window = int(len_data/window_length)
            for w in range(n_window):
                for electrode in range(n_col):
                    segment = data_read[electrode][w*window_length:w*window_length+window_length]

                    data[w+count][electrode] = segment
                    mav_data[w+count][electrode] = self.getMAV(segment)
                    rms_data[w+count][electrode] = self.getRMS(segment)
                    var_data[w+count][electrode] = self.getVAR(segment)
                    sd_data[w+count][electrode] = self.getSD(segment)
                target.append(int(file[0:3]))
            count += w + 1
        print(target)
        mav_data = preprocessing.scale(mav_data)
        rms_data = preprocessing.scale(rms_data)
        var_data = preprocessing.scale(var_data)
        sd_data = preprocessing.scale(sd_data)

        self.emg_data = {'data' : data, 'target' : target, 'mav' : mav_data, 'rms' : rms_data, 'var' : var_data, 'sd' : sd_data}


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


    def plot_hitogram_mvmnts(self):
        """ Affiche un histogramme représentant la répartition des mouvements provenant du dataset"""
        target = self.emg_data.get('target')
        all_classes = np.unique(target)
        classes_count = np.zeros(np.size(all_classes))
        fig, subfig = plt.subplots()
        bins = []
        for i in all_classes:
            bins.append(i - 0.25)
            bins.append(i + 0.25)
        for i in target:
            classes_count[i] += 1
        subfig.hist(all_classes, bins=bins, weights=classes_count)
        subfig.set_xlabel('Classe du mouvement')
        subfig.set_ylabel("Nombre d'occurences [-]")
        subfig.set_title(f'''Histogramme représentant le nombre de d'occurances de chaque classe dans {self.path.split('/')[1]}''')
        plt.show() 


    def plot_jeu_2_electrodes(self, ch0=6, ch1 = 17, classes='all', legend_with_name=False):
        target = self.emg_data.get('target')
        if classes == 'all':
            classes = np.unique(target)
        else:
            classes = np.array(classes)
        pairs = [(i, j) for i in range(2) for j in range(2)]
        stats = ['mav', 'rms', 'var', 'sd']
        fig, subfigs = plt.subplots(2, 2, tight_layout=True)
        for count, (f1, f2) in enumerate(pairs):
            for c in classes:
                ind = np.where(target == c)
                if legend_with_name is True:
                    label = f'{c} : {self.int_to_mvmnt_csv(c)}'
                else:
                    label = f'Class : {c}'
                subfigs[(f1, f2)].scatter(self.emg_data.get(stats[count])[ind, ch0], self.emg_data.get(stats[count])[ind, ch1], label=label)
                subfigs[(f1, f2)].set_xlabel(f'Channel {ch0}')
                subfigs[(f1, f2)].set_ylabel(f'Channel {ch1}')
                subfigs[(f1, f2)].set_title(f'Selected feature : {stats[count]}')
                subfigs[(f1, f2)].legend()
        plt.show()


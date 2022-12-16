import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import json
from useful_functions import printProgressBar
import time

# Jeux de données
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Classifieurs
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

plt.rcParams.update({'font.size': 16})

class DataProcessing:
    """
    Classe qui va traiter les électromyogrammes en fonction du type de fichier qu'il reçoit.
    Dans ce projet, on va traiter des électromyogrammes venant de deux sources différentes, qui ont sauvegardés leurs signaux de façon différente (csv ou .mat)
    Dans les deux cas, on va traiter les données et les reformat pour produire un fichier .txt, dans lequel on va sauvegarder un dictionnaire.
    Celui-ci va contenir notre jeu et nos targets dans un format que l'on est plus habitué (avec array numpy ou tenseurs pytorch)
    Autres points à aborder : 
    - Est-ce que avec Capgmyo on pourrait mettre tous les sujets ensemble ? est-ce que ça donnerait un meilleur résultat ?
    - Est-ce qu'on a vraiment besoin de sauvegarder data ? c'est juste utile pour plot emg signal, et ça sert pas vraiment à grand chose
    - Est-ce que les datas de Capgmyo sont vraiment ok ?
    """

    # Variables de classe
    sample_frequency = 1000
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']

    """constructeur pourc initialiser le path et le type de fichier (.csv ou .mat)"""
    def __init__(self, path, f_types):
        self.path = path
        self.f_types = f_types

    """
    #Computes the Mean Absolute Value (MAV)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Mean Absolute Value as [float]
    """
    def findMeanAbsoluteValue(self, x, axis=None):
        MAV = np.mean(np.abs(x), axis)
        return MAV
    

    """
    Computes the Root Mean Square value (RMS)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Root Mean Square value as [float]
    """
    def findRootMeanSquareValue(self, x, axis=None):
        RMS = np.sqrt(np.mean(x**2, axis))
        return RMS
    
    """
    Computes the Variance of EMG (Var)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Variance of EMG as [float]
    """
    def findVariance(self, x, axis=None):
        N = np.shape(x)[-1]
        Var = (1/(N-1))*np.sum(x**2, axis)
        return Var
    
    """
    Computes the Standard Deviation (SD)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Standard Deviation as [float]
    """
    def findStandardDerivation(self, x, axis=None):
        N = np.shape(x)[-1]
        xx = np.mean(x, axis)
        SD = np.sqrt(1/(N-1)*np.sum((x-xx)**2, axis))
        return SD

    """description"""
    def intToMovementCSV(self, mvmnt):
        assert self.f_types == 'csv'
        mvmnts = {0: 'Closed fist', 1: 'Thumb up', 2: 'Tri-pod pinch', 3: 'Neutral hand position', 4: 'Fine pinch (index+thumb)', 5: 'Pointed index'}
        return mvmnts.get(mvmnt)
    
    """ format csv files and makes sure files extention is .csv """
    def formatCSVFiles(self, window_length=200):
        assert self.f_types == 'csv'
        subject = '0'
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
                    mav_data[w+count][electrode] = self.findMeanAbsoluteValue(segment)
                    rms_data[w+count][electrode] = self.findRootMeanSquareValue(segment)
                    var_data[w+count][electrode] = self.findVariance(segment)
                    sd_data[w+count][electrode] = self.findStandardDerivation(segment)
                target.append(int(file[0:3]))
            count += w + 1

        mav_data = preprocessing.scale(mav_data)
        rms_data = preprocessing.scale(rms_data)
        var_data = preprocessing.scale(var_data)
        sd_data = preprocessing.scale(sd_data)

        self.emg_data = {subject : {'data' : data, 'target' : target, 'mav' : mav_data, 'rms' : rms_data, 'var' : var_data, 'sd' : sd_data}}

    """ format mat files and makes sure files extention is .mat """
    def formatMATFiles(self, window_length=150, name_of_txt_file='first_data_set_', overwrite=False):
        assert self.f_types == 'mat'

        path_levels = self.path.split('/')
        save_path = f'{path_levels[0]}/{path_levels[1]}/txt_files/{name_of_txt_file}{str(window_length)}.txt'
        if overwrite is False and os.path.exists(save_path) is True:
            print(f"The file {name_of_txt_file}{str(window_length)} already exists, no need to format")
            self.emg_data = json.load(open(save_path))
        if overwrite is True or os.path.exists(save_path) is False:

            emg_data = {}
            list_of_data = os.listdir(self.path)
            printProgressBar(0, len(list_of_data), prefix = f'Format in progress : {name_of_txt_file}{str(window_length)}', suffix= 'Complete', length=50)
            for i in range(len(list_of_data)):

                # progress bar
                printProgressBar(i+1, len(list_of_data), prefix = f'Format in progress : {name_of_txt_file}{str(window_length)}', suffix= 'Complete', length=50)
                subject = str(int(list_of_data[i][0:3]))
                mvmnt = int(list_of_data[i][4:7])
                if emg_data.get(subject) == None:
                    emg_data[subject] = {'data' : [], 'mav' : [], 'rms' : [], 'var' : [], 'sd' : [], 'target' : []} 

                # TO DO : ajouter la semgmentation des signaux
                emg_signals = scipy.io.loadmat(self.path + '/' + list_of_data[i]).get('data')

                n_col = np.shape(emg_signals)[1]

                n_window = int(np.shape(emg_signals)[0]/window_length)
                for w in range(n_window):
                    data, mav_data, rms_data, var_data, sd_data = [], [], [], [], []
                    for electrode in range(n_col):
                        segment = emg_signals[electrode][w*window_length:w*window_length+window_length]
                        data += [segment.tolist()]

                        mav_data += [self.findMeanAbsoluteValue(segment).tolist()]
                        rms_data += [self.findRootMeanSquareValue(segment).tolist()]
                        var_data += [self.findVariance(segment).tolist()]
                        sd_data += [self.findStandardDerivation(segment).tolist()]

                    if np.isnan(mav_data).any():
                        continue

                    emg_data[subject]['mav'].append(mav_data)
                    emg_data[subject]['rms'].append(rms_data)
                    emg_data[subject]['var'].append(var_data)
                    emg_data[subject]['sd'].append(sd_data)
                    emg_data[subject]['data'].append(data)
                    emg_data[subject]['target'].append(mvmnt)


            self.emg_data = emg_data

            with open(save_path, 'w+') as data_file:
                data_file.write(json.dumps(emg_data))
        
    """ Description """
    def normalizeSet(self):
        to_norm = ['mav', 'rms', 'var', 'sc']
        for subject, dict_data in self.emg_data.items():
            for stat, data in dict_data.items():
                if stat in to_norm:
                    self.emg_data[subject][stat] = preprocessing.scale(data)
    
    """ Split dataset """
    def createTrainingSet(self, subject, feature='rms', test_size=0.1, random_state=42):
        self.feature = feature
        x_data = self.emg_data[subject][feature]
        y_data = self.emg_data[subject]['target']
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
        return (X_train, X_test, y_train, y_test)

    """ Implémenter la méthode des k plus proche voisins (Devoir 2 #3) """ 
    def classifierKPlusProcheVoisins(self, subject, feature='rms', k=2, weigths_param = 'uniform'):
        emg_data = preprocessing.minmax_scale(self.emg_data[subject][feature])
        target = np.array(self.emg_data[subject]['target'])

        loo = LeaveOneOut()
        loo.get_n_splits(emg_data)

        score = 1
        for train_index, test_index in loo.split(emg_data):
            X_train, X_test = emg_data[train_index], emg_data[test_index]
            y_train, y_test = target[train_index], target[test_index]

            neigh = KNeighborsClassifier(n_neighbors=k, weights=weigths_param)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)

            if y_pred != y_test:
                score -= 1/np.size(target)
        score *= 100
        print(f'classifier_k_plus_proche_voisins avec k = {k} et weights = {weigths_param}. Score : {np.round(score, 1)}%')

    """ Affiche une figure contenant le signal emg à gauche et sa transformée de fourier à droite """
    def plotEMGSignalAndFFT(self, emg_signal):
        # to do : axes : temps, intensité, fréquences, intensité
        ps = np.abs(np.fft.fft(emg_signal))**2
        time_step = 1/self.sample_frequency
        freqs = np.fft.fftfreq(np.size(emg_signal), time_step)
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

    """ Affiche un histogramme représentant la répartition des mouvements provenant du dataset"""
    def plotMovementHistogram(self, subject):
        target = self.emg_data[subject].get('target')
        all_classes = np.unique(target)
        classes_count = {}
        fig, subfig = plt.subplots()
        bins = []
        x_coordinates = np.arange(np.size(all_classes))
        for i in x_coordinates:
            bins.append(i - 0.25)
            bins.append(i + 0.25)
        for i in target:
            if classes_count.get(i) is None:
                classes_count[i] = 1
            else:
                classes_count[i] += 1
        plt.setp(subfig, xticks=x_coordinates, xticklabels=all_classes)
        subfig.bar(x_coordinates, classes_count.values(), width=0.8)
        subfig.set_xlabel('Classe du mouvement')
        subfig.set_ylabel("Nombre d'occurences [-]")
        subfig.set_xticklabels(all_classes)

        subfig.set_title(f'''Histogramme représentant le nombre de d'occurances de chaque classe dans {self.path.split('/')[1]}''')
        plt.show() 


    def plot2ElectrodeSet(self, subject, ch0=6, ch1 = 17, classes='all', legend_with_name=False):
        target = self.emg_data[subject].get('target')
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
                    label = f'{c} : {self.intToMovementCSV(c)}'
                else:
                    label = f'Class : {c}'
                x = np.array(self.emg_data[subject].get(stats[count]))[ind, ch0]
                y = np.array(self.emg_data[subject].get(stats[count]))[ind, ch1]
                subfigs[(f1, f2)].scatter(x, y, label=label)
                subfigs[(f1, f2)].set_xlabel(f'Electrode #{ch0}')
                subfigs[(f1, f2)].set_ylabel(f'Electrode #{ch1}')
                subfigs[(f1, f2)].set_title(f'Selected feature : {stats[count]}')
                subfigs[(f1, f2)].legend()
        plt.show()

    """ Apply parametric classifier on dataset, plot for electrode ch0 and ch1 """
    def plot2ElectrodeParametricClassifier(self, data_set, ch0=6, ch1=17, classes='all', legend_with_name=False):
        classifiers = [QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(), GaussianNB(), NearestCentroid()]

        h = 0.02
        x_min, x_max = data_set[0][:, ch0].min() - 1, data_set[0][:, ch0].max() + 1
        y_min, y_max = data_set[0][:, ch1].min() - 1, data_set[0][:, ch1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        data = np.c_[data_set[0][:, ch0], data_set[0][:, ch1]]

        if classes == 'all':
            classes = np.unique(data_set[2])
            target = np.array(data_set[2])
        if classes != 'all':
            classes = np.array(classes)
            ind = np.isin(data_set[2], classes)
            data = data[ind]
            target = np.array(data_set[2])[ind]

        fig, subfigs = plt.subplots(2, 2, sharex='all', sharey='all', tight_layout=True)

        for clf, subfig in zip(classifiers, subfigs.reshape(-1)):
            clf_name = clf.__class__.__name__
            clf.fit(data, target)
            Y = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Y = Y.reshape(xx.shape)

            # déterminer les y pas ok avec predict, et ceux qui sont pas ok, scatter, gros rond rouge
            # ne sert par à calculer le score. Score doit être testé avec le jeu de test, non le jeu d'entraînement
            y_pred = clf.predict(data)
            bad_classification = np.argwhere(target - y_pred != 0)

            subfig.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.5)
            subfig.scatter(data[bad_classification, 0], data[bad_classification, 1], s=100, facecolors='none', edgecolors='r')   
            colour_count = 0
            for c in classes:
                ind = np.where(target == c)
                if legend_with_name is True:
                    label = f'{c} : {self.intToMovementCSV(c)}'
                else:
                    label = f'Class : {c}'
                subfig.scatter(data[ind, 0], data[ind, 1], c=self.colors[colour_count], label=label)
                subfig.set_xlabel(f'Electrode #{ch0}')
                subfig.set_ylabel(f'Electrode #{ch1}')
                subfig.set_title(f'{clf_name} with {self.feature}')
                subfig.legend()
                colour_count += 1

        plt.show()

    """ Function that calls previous plot fonctions """
    def plotForBothDatasetsVerification(self, subject, feature, k, ch0, ch1, classes):
        self.classifierKPlusProcheVoisins(subject=subject, feature=feature, k=k)
        self.plotEMGSignalAndFFT(self.emg_data[subject].get('data')[0][0])
        self.plotMovementHistogram(subject)
        self.plot2ElectrodeSet(subject=subject, classes=classes, ch0=ch0, ch1=ch1)

        data_set = self.createTrainingSet(subject=subject, feature=feature)

        self.plot2ElectrodeParametricClassifier(data_set=data_set, ch0=ch0, ch1=ch1, classes=classes, legend_with_name=False)


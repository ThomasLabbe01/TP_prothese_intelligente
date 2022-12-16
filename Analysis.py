from DataProcessing import DataProcessing
from Classifications import Classifications
import numpy as np
import matplotlib.pyplot as plt
from useful_functions import printProgressBar


""" Calculer et afficher la précision de tous les classements en fonction du temps de traitement pour un même feature
        
    Cette fonction va créer un graphique : score vs 
    n_window pour les postures choisies
"""
def calculate_and_plot_score_vs_window(accessPath, fileType, window_length, plot_figure = False):
    scores = []
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
    for w in window_length:
        emgCSV = DataProcessing(accessPath, fileType)
        emgCSV.formatCSVFiles(window_length=w)
        emgCSV.normalizeSet()
    
        classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=w)
        classifications.data_segmentation(method='train_test_split', proportions=[0.75, 0.25, 0])
        predictions = classifications.classifieurMethodeDeVote()
        classifications.calculate_general_score(predictions)
        classScore = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = False)
        scores.append(classScore.diagonal())
    scores = np.array(scores)
    
    if plot_figure == True:
        for c in range(np.shape(scores)[1]):
            plt.plot(window_length, scores[:, c], color=colors[c], label=f'Posture {c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Score [-]")
        plt.title("Précision du classifieur en fonction du temps d'acquisition pour différents mouvements de mains \n Classifieur : {:s}".format(classifications.clfName))
        plt.show()
    
    return scores

""" Calculer et afficher la précision d'un classement avec une certaine statistique en fonction du temps de traitement 
        
    Cette fonction va créer un graphique : score vs 
    n_window pour les features choisis
"""
def calculate_and_plot_score_vs_feature(accessPath, fileType, window_length, posture=2, plot_figure = False):
    scores = {'mav' : [], 'rms' : [], 'var' : [], 'sd' : []}
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
    statistiques = ['mav', 'rms', 'var', 'sd']

    printProgressBar(0, len(window_length), prefix = 'calculate_and_plot_score_vs_feature in progress', suffix= 'Complete', length=50)
    for index, w in enumerate(window_length):
        emgCSV = DataProcessing(accessPath, fileType)
        emgCSV.formatCSVFiles(window_length=w)
        emgCSV.normalizeSet()
        for s in statistiques:
            classifications = Classifications(emgCSV.emg_data, subject='0', statistique=s, window_length=w)
            classifications.data_segmentation(method='train_test_split', proportions=[0.75, 0.25, 0])
            predictions = classifications.classifierKPlusProcheVoisins()
            classifications.calculate_general_score(predictions)
            classScore = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = False)
            scores[s].append(classScore.diagonal())
        printProgressBar(index+1, len(window_length), prefix = 'calculate_and_plot_score_vs_feature in progress', suffix= 'Complete', length=50)

    if plot_figure == True:
        for index, c in enumerate(statistiques):
            scores_c = np.array(scores[c])[:, posture]  # numéro de la posture
            plt.plot(window_length, scores_c, color=colors[index], label=f'{c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Score [-]")
        plt.title("Précision du classifieur en fonction du temps d'acquisition, pour différentes statistiques mesurées \n Classifieur : {:s}, Posture : {}".format(classifications.clfName, posture))
        plt.show()
    
    return scores

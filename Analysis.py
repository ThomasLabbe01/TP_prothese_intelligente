from DataProcessing import DataProcessing
from Classifications import Classifications
import numpy as np
import matplotlib.pyplot as plt
from useful_functions import printProgressBar


""" Calculer et afficher la précision de tous les classements en fonction du temps de traitement pour un même feature
        
    Cette fonction va créer un graphique : score vs 
    n_window pour les postures choisies
"""
def calculate_and_plot_score_vs_window(accessPath, fileType, window_length, plot_score = False, plot_measurements = False):
    scores = []
    badMeasurements = []
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
    printProgressBar(0, len(window_length), prefix = 'calculate_and_plot_score_vs_window in progress', suffix= 'Complete', length=50)
    for index, w in enumerate(window_length):
        if fileType == 'csv':
            emgCSV = DataProcessing(accessPath, fileType)
            emgCSV.formatCSVFiles(window_length=w)
            emgCSV.normalizeSet()
            classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=w)
            datasetName = 'GEL-4072'
        if fileType == 'mat':
            emgMAT = DataProcessing(accessPath, fileType)
            emgMAT.formatMATFiles(window_length=w, name_of_txt_file = 'first_data_set_', overwrite = False)
            emgMAT.normalizeSet()
            classifications = Classifications(emgMAT.emg_data, subject='1', statistique='mav', window_length=w)
            datasetName = 'Capgmyo'

        classifications.data_segmentation(method='train_test_split', proportions=[0.75, 0.25, 0])
        predictions = classifications.classifieurMethodeDeVote()
        classifications.calculate_general_score(predictions)
        classScore, classBadMeasurements = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = False)
        scores.append(classScore.diagonal())
        badMeasurements.append(classBadMeasurements.diagonal())
        printProgressBar(index+1, len(window_length), prefix = 'calculate_and_plot_score_vs_window in progress', suffix= 'Complete', length=50)

    scores = np.array(scores)
    badMeasurements = np.array(badMeasurements)
    
    if plot_score == True:
        # Le fait d'avoir un plus grand sample pour des n_window plus petit fait en sorte que l'erreur relative est biaisée
        for c in range(np.shape(scores)[1]):
            if c > 5:
                continue
            plt.plot(window_length, scores[:, c], color=colors[c], label=f'Posture {c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Score [-]")
        plt.title("Précision du classifieur en fonction du temps d'acquisition pour différents mouvements de mains \n Dataset : {:s}, Classifieur : {:s}".format(datasetName, classifications.clfName))
        plt.show()

    if plot_measurements == True:
        for c in range(np.shape(scores)[1]):
            if c > 5:
                continue
            plt.plot(window_length, badMeasurements[:, c], color=colors[c], label=f'Posture {c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Nombre de mouvements mal classés [-]")
        plt.title("Erreur absolue du classifieur en fonction du temps d'acquisition pour différents mouvements de mains \n Dataset : {:s}, Classifieur : {:s}".format(datasetName, classifications.clfName))
        plt.show()

    return scores

""" Calculer et afficher la précision d'un classement avec une certaine statistique en fonction du temps de traitement 
        
    Cette fonction va créer un graphique : score vs 
    n_window pour les features choisis
"""
def calculate_and_plot_score_vs_feature(accessPath, fileType, window_length, posture=2, plot_score = False, plot_measurements = False):
    scores = {'mav' : [], 'rms' : [], 'var' : [], 'sd' : []}
    badMeasurements = {'mav' : [], 'rms' : [], 'var' : [], 'sd' : []}
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
    statistiques = ['mav', 'rms', 'var', 'sd']

    printProgressBar(0, len(window_length), prefix = 'calculate_and_plot_score_vs_feature in progress', suffix= 'Complete', length=50)
    for index, w in enumerate(window_length):
        if fileType == 'csv':
            emgCSV = DataProcessing(accessPath, fileType)
            emgCSV.formatCSVFiles(window_length=w)
            emgCSV.normalizeSet()
            datasetName = 'GEL-4072'
            
        for s in statistiques:
            classifications = Classifications(emgCSV.emg_data, subject='0', statistique=s, window_length=w)
            classifications.data_segmentation(method='train_test_split', proportions=[0.75, 0.25, 0])
            predictions = classifications.classifierKPlusProcheVoisins()
            classifications.calculate_general_score(predictions)
            classScore, classBadMeasurements = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = False)
            scores[s].append(classScore.diagonal())
            badMeasurements[s].append(classBadMeasurements.diagonal())
        printProgressBar(index+1, len(window_length), prefix = 'calculate_and_plot_score_vs_feature in progress', suffix= 'Complete', length=50)

    if plot_score == True:
        # Le fait d'avoir un plus grand sample pour des n_window plus petit fait en sorte que l'erreur relative est biaisée
        for index, c in enumerate(statistiques):
            scores_c = np.array(scores[c])[:, posture]  # numéro de la posture
            plt.plot(window_length, scores_c, color=colors[index], label=f'{c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Score [-]")
        plt.title("Précision du classifieur en fonction du temps d'acquisition, pour différentes statistiques mesurées \n Dataset : {:s} Classifieur : {:s}, Posture : {}".format(datasetName, classifications.clfName, posture))
        plt.show()
    
    if plot_measurements == True:
        for index, c in enumerate(statistiques):
            badMeasurements_c = np.array(badMeasurements[c])[:, posture]  # numéro de la posture
            plt.plot(window_length, badMeasurements_c, color=colors[index], label=f'{c}', linewidth=3, marker='o')

        plt.legend()
        plt.xlabel("Temps d'acquisition [ms]")
        plt.ylabel("Score [-]")
        plt.title("Erreur absolue du classifieur en fonction du temps d'acquisition, pour différentes statistiques mesurées \n Dataset : {:s} Classifieur : {:s}, Posture : {}".format(datasetName, classifications.clfName, posture))
        plt.show()
    
    return scores

from DataProcessing import DataProcessing
from Classifications import Classifications
import numpy as np
import matplotlib.pyplot as plt

def calculate_and_plot_score_vs_window(accessPath, fileType, window_length, plot_figure = False):
    scores = []
    colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
    for w in window_length:
        emgCSV = DataProcessing(accessPath, fileType)
        emgCSV.formatCSVFiles(window_length=w)
        emgCSV.normalizeSet()
    
        classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav')
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

class Analysis:

    """ Calculer et afficher la précision d'un classement en fonction du temps de traitement 
        Il serait mieux de créer cette fonction dans useful_functions, de faire une boucle qui va recréer un dataset avec n_window
        et de recalculer le score avec la méthode au choix
        
        Cette fonction va créer un graphique : score vs n_window pour les méthodes choisis
        Cette fonction va être un scatter plot, et la taille de chaque point va être proportionnelle à la quantité de signaux qu'on a
    """
    def calculate_and_plot_score_vs_window(self):
        return

    """ Calculer et afficher la précision d'un classement en fonction du temps de traitement 
        Il serait mieux de créer cette fonction dans useful_functions, de faire une boucle qui va recréer un dataset avec n_window
        et de recalculer le score avec le feature au choix
        
        Cette fonction va créer un graphique : score vs n_window pour les features choisis"""
    def calculate_and_plot_score_vs_feature(self):
        return

    
        """ Calculer et afficher la précision d'un classement en fonction du temps de traitement 
        Il serait mieux de créer cette fonction dans useful_functions, de faire une boucle qui va recréer un dataset avec n_window
        et de recalculer le score avec le feature au choix
        
        Cette fonction va créer un graphique : score vs n_window pour les features choisis"""
    def calculate_and_plot_score_vs_subject_with_best_feature(self):
        return
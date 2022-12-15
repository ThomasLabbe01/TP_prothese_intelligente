from mlxtend.plotting import plot_confusion_matrix
from DataProcessing import DataProcessing
from Classifications import Classifications
import numpy as np
import matplotlib.pyplot as plt

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
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold, train_test_split
from Eletromyogram_analysis import Electromyogram_analysis

class Classifications:
    """ Classe qui hérite de Electromyogram_analysis. Dans cette classe, on va définir toutes les fonctions qui font de la classification avec tous les électrodes
        On va également définir des fonctions qui vont segmenter le jeu de données pour utiliser le même dans toutes les classes, et des fonctions qui calculent les scores
        On veut également implémenter les méthodes suivantes : 
        - Méthodes de vote : Prendre les 4 classifieurs paramétriques + knn ?
        - Bagging : Ensemble de classifieurs entraînées sur ensembles de données légèrement différents
        - Autres ... """
    def __init__(self, data):
        """init class pour emg_data"""
        self.data = data
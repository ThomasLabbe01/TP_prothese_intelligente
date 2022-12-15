import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold, train_test_split, LeaveOneOut
from DataProcessing import DataProcessing

class Classifications:
    """ Classe qui hérite de Electromyogram_analysis. Dans cette classe, on va définir toutes les fonctions qui font de la classification avec tous les électrodes
        On va également définir des fonctions qui vont segmenter le jeu de données pour utiliser le même dans toutes les classes, et des fonctions qui calculent les scores
        Autres choses à implémenter : 
        - Bagging : Ensemble de classifieurs entraînées sur ensembles de données légèrement différents
        - Vérifier s'il y aurait pas du prétraitement à faire. Est-ce qu'on est capable d'enlever des électrodes pour améliorer le score ? Bien que c'est un peu l'idée du réseau à convolution (sélection avant et arrière séquentielle
        - Boosting avec petit n_window ? c'est possible ?
        
        Réseau à convolution devrait être la méthode qui nous permet d'obtenir le meilleur score possible
        Quelle est l'architecture qu'on cherche à avoir ?
        Quelle est l'architecture qui est utilisée par le groupe de l'université Laval ?
        Est-ce qu'on peut mettre tous les datas des sujets ensemble ? 
        Est-ce qu'on peut faire du transfert de représentation pour réutiliser un réseau existant pour un sujet, et l'appliquer à un autre ?
        TO DO : https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
        """
    def __init__(self, data, subject, statistique):
        """init class pour emg_data"""
        self.data = data[subject]
        self.subject = subject
        self.colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
        self.statistique = statistique

    """ Fonction qui va ségmenter le jeu de données selon proportions.
        Si validation = True, size(proportions) = 3, [train, test, validation] 
        Implémenter les méthodes suivantes : 
        - train_test_split
        - RepeatedKFolder
        - KFold
        - LeaveOneOut (peut-être plus approprié de définir dans Knn

        Cette fonction va créer (X_train, X_test, X_validation), (y_train, y_test, y_validation)
    """
    def data_segmentation(self, method, proportions= [0.7, 0.15, 0.15]):
        assert len(proportions) == 3, 'La variable proportions doit avoir une taille de 3 : [entraînement, test, validation]. Si on veut une taille nulle au jeu de validation, validation = 0'
        trainData = []
        testData = []
        validationData = []
        if method == 'train_test_split':
            if proportions[2] != 0:
                X_train, X_test, y_train, y_test = train_test_split(self.data[self.statistique], self.data['target'], test_size = proportions[1]+proportions[2], random_state=42)
                X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size = proportions[2]/(proportions[1]+proportions[2]), random_state=42)
                trainData = [X_train, y_train]
                testData = [X_test, y_test]
                validationData = [X_validation, y_validation]

            if proportions[2] == 0:
                X_train, X_test, y_train, y_test = train_test_split(self.data[self.statistique], self.data['target'], test_size = proportions[1], random_state=42)
                trainData = [X_train, y_train]
                testData = [X_test, y_test]
        
        self.trainData = trainData
        self.testData = testData
        self.validationData = validationData

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

    """ Fonction qui va calculer le score d'un classifieur.
        Cette fonction retourne un score sur 100
    """ 
    def calculate_score(self):
        return

    """ Fonction qui va calculer le score de chaque mouvement dans le classifieur
    """
    def calculate_score_par_classe(self):
        return

    """ Fonction qui va faire une classification avec la méthode paramétrique NearestCentroid()
        Cette fonction retourne y_pred
    """
    def classifieur_nearest_centroid(self):
        clf = NearestCentroid()
        clf.fit(self.trainData[0], self.trainData[1])

        return clf.predict(self.testData[0]) 

    """ Fonction qui va faire une classification avec la méthode paramétrique GaussianNB()
        Cette fonction retourne y_pred
    """
    def classifieur_noyau_gaussien(self):
        clf = GaussianNB()
        clf.fit(self.trainData[0], self.trainData[1])

        return clf.predict(self.testData[0])
    
    """ Fonction qui va faire une classification avec la méthode paramétrique LinearDiscriminantAnalysis()
        Cette fonction retourn y_pred
    """
    def classifieur_lineaire(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.trainData[0], self.trainData[1])

        return clf.predict(self.testData[0])

    """ Fonction qui va faire une classification avec la méthode paramétrique QuadraticDiscriminantAnalysis()
        Cette fonction retourne y_pred
    """
    def classifeur_quadratique(self):
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(self.trainData[0], self.trainData[1])

        return clf.predict(self.testData[0])
    
    """ Implémenter un classifier linéaire avec svm """
    def classifieur_lineaire_svm(self):
        return

    """ Implémenter l'option de rejet avec le nearest centroid (Devoir 1, #3) """
    def classifier_nearest_centroid_avec_option_de_rejet(self):
        return

    """ Implémenter la méthode des k plus proche voisins (Devoir 2 #3) """
    def classifier_k_plus_proche_voisins(self, k=3, weights_param='uniform'):
        data = np.array(self.data[self.statistique])
        target = np.array(self.data['target'])
        loo = LeaveOneOut()
        loo.get_n_splits(data)
        print(target)
        y_pred = []
        for train_index, test_index in loo.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, _ = target[train_index], target[test_index]

            neigh = KNeighborsClassifier(n_neighbors=k, weights=weights_param)
            neigh.fit(X_train, y_train)
            y_pred.append(neigh.predict(X_test)[0])
        
        return y_pred

    """ Implémenter la méthode de vote avec les méthodes suivantes : 
        - classifieur_nearest_centroid
        - classifieur_noyau_gaussien
        - classifieur_lineaire
        - classifieur_quadratique
        - classifieur_k_nearest_neighbour
        1. aller chercher y_pred des classifieurs ci-haut
        2. prendre le y_pred le plus fréquent
        3. comparer à target
        4. calculer le score
    """
    def methode_de_vote(self):
        return

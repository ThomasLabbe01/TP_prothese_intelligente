import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold, train_test_split, LeaveOneOut
from DataProcessing import DataProcessing
from mlxtend.plotting import plot_confusion_matrix

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
    def __init__(self, data, subject, statistique, window_length):
        """init class pour emg_data"""
        self.data = data[subject]
        self.subject = subject
        self.colors = ['#120bd6', '#00d600', '#Ff0000', '#Ffb300', '#Ff5900', '#541c00']
        self.statistique = statistique
        self.clfName = ''
        self.window_length = window_length

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

    """ Fonction qui va calculer le score d'un classifieur.
        Cette fonction retourne un score sur 100
    """ 
    def calculate_general_score(self, predict_data):
        good_predictions = np.where(predict_data - self.testData[1] == 0)
        self.totalScore = 100 * np.size(good_predictions)/np.size(predict_data)

    """ Fonction qui va calculer le score de chaque mouvement dans le classifieur
    """
    def matriceDeConfusion(self, predict_data, plot_figure = False):
        assert self.clfName != '', "Il faut d'abord faire une classification avec une méthode qui commence par 'classifieur'"
        change_results = 0
        target = np.array(self.testData[1])
        classes = np.unique(target)
        if np.min(classes) == 1:
            classes = classes - 1  # Avec le dataset capgmyo, les classes vont de 1 à 8, ce qui fait un bug avec l'autre dataset qui va de 0 à 5
            predict_data = predict_data - 1
            change_results = 1
        postures = [f'Posture {i}' for i in classes]
        classScore = np.zeros((np.size(classes), np.size(classes)))
        classBadMeasurements = np.zeros((np.size(classes), np.size(classes)))
        for c in classes:
            class_c_index = np.where(predict_data == c)
            results = target[class_c_index]
            mouvement, count = np.unique(results, return_counts=True)
            if change_results == 1:
                mouvement -= 1
            for index, mvmnt in enumerate(mouvement):
                classScore[c, mvmnt] = count[index]
                if c != mvmnt:
                    classBadMeasurements[c, c] += count[index]
            if np.size(count) == 0:  # Si on a trouvé aucun point qui correspond à la classe c, car les classifieurs ne sont pas adéquats
                classScore[c, :] *= 1
            else:
                classScore[c, :] *= 1/np.sum(classScore[c, :])

        if plot_figure == True:
            plt.rcParams.update({'font.size': 13})
            fig, ax = plot_confusion_matrix(conf_mat=classScore,
                                            colorbar=True,
                                            show_absolute=False,
                                            show_normed=True,
                                            class_names=postures)
            plt.title("Normalized confusion matrix with {:s}, Score total = {:.2f} % \n Temps d'acquisiton : {} ms".format(self.clfName, self.totalScore, self.window_length))
            plt.show()   
        return classScore, classBadMeasurements


    """ Fonction qui va faire une classification avec la méthode paramétrique NearestCentroid()
        Cette fonction retourne y_pred
    """
    def classifieurNearestCentroid(self):
        clf = NearestCentroid()
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurNearestCentroid'
        return clf.predict(self.testData[0]) 

    """ Fonction qui va faire une classification avec la méthode paramétrique GaussianNB()
        Cette fonction retourne y_pred
    """
    def classifieurNoyauGaussien(self):
        clf = GaussianNB()
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurNoyauGaussien'
        return clf.predict(self.testData[0])
    
    """ Fonction qui va faire une classification avec la méthode paramétrique LinearDiscriminantAnalysis()
        Cette fonction retourn y_pred
    """
    def classifieurLineaire(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurLineaire'
        return clf.predict(self.testData[0])

    """ Fonction qui va faire une classification avec la méthode paramétrique QuadraticDiscriminantAnalysis()
        Cette fonction retourne y_pred
    """
    def classifeurQuadratique(self):
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifeurQuadratique'
        return clf.predict(self.testData[0])
    
    """ Implémenter un classifier linéaire avec svm """
    def classifieurLineaire(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurLineaire'
        return clf.predict(self.testData[0])

    """ Implémenter l'option de rejet avec le nearest centroid (Devoir 1, #3) """
    def classifierNearestCentroidAvecOptionDeRejet(self):
        return

    """ Implémenter la méthode des k plus proche voisins (Devoir 2 #3) """
    def classifierKPlusProcheVoisins(self, k=3, weights_param='uniform'):
        neigh = KNeighborsClassifier(n_neighbors=k, weights=weights_param)
        neigh.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifierKPlusProcheVoisins'
        return neigh.predict(self.testData[0])


    """ Implémenter la méthode sklearn.tree.DecisionTreeClassifier"""
    def classifieurDecisionTree(self, max_depth=5):
        clf = DecisionTreeClassifier(max_depth = max_depth)
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurDecisionTree'
        return clf.predict(self.testData[0])

    """ Implémenter la méthode sklearn.ensemble.RandomForestClassifier"""
    def classifieurRandomDecisionTree(self, max_dept=5, n_estimator=10, max_features=1):
        clf = RandomForestClassifier(max_depth=max_dept, n_estimators=n_estimator, max_features=max_features)
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurRandomDecisionTree'
        return clf.predict(self.testData[0])

    """ Implémenter la méthode sklearn.ensemble.AdaBoostClassifier"""
    def classifieurAdaBoost(self, n_estimators=100, random_state=42):
        clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(self.trainData[0], self.trainData[1])

        self.clfName = 'classifieurAdaBoost'
        return clf.predict(self.testData[0])


    """ Implémenter la méthode de vote avec les méthodes suivantes : 
        - classifieur_nearest_centroid
        - classifieur_noyau_gaussien
        - classifieur_lineaire
        - classifieur_quadratique
        - classifieur_k_nearest_neighbour
        - classifieurDecisionTree
        - classifieurRandomDecisionTree
        - classifieurAdaBoost
        1. aller chercher y_pred des classifieurs ci-haut
        2. prendre le y_pred le plus fréquent
        3. comparer à target
        4. calculer le score
    """
    def classifieurMethodeDeVote(self):
        classifieurs = np.array([self.classifieurNearestCentroid(),
                                 self.classifieurLineaire(), 
                                 self.classifieurNoyauGaussien(),
                                 self.classifeurQuadratique(), 
                                 self.classifierKPlusProcheVoisins(k=3, weights_param='uniform'), 
                                 self.classifieurDecisionTree(max_depth=5), 
                                 self.classifieurRandomDecisionTree(max_dept=5, n_estimator=10, max_features=1), 
                                 self.classifieurAdaBoost(n_estimators=100, random_state=42)])
        
        voteResults = []
        for i in range(np.shape(classifieurs)[1]):
            counts = np.bincount(classifieurs[:, i])
            voteResults.append(np.argmax(counts))
        self.clfName = 'classifieurMethodeDeVote'
        return np.array(voteResults)

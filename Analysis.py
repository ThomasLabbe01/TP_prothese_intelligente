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
"""
## load data
def loadData():
    accessPath = 'all_data/data_2_electrode_GEL_4072'
    fileType = 'csv'
    window_length = 75
    emgCSV = DataProcessing(accessPath, fileType)
    emgCSV.formatCSVFiles(window_length=window_length)
    emgCSV.normalizeSet()
    classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
    classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
"""

## Optimisation
def optimisationClassifieurKPPV():
    accessPath = 'all_data/data_2_electrode_GEL_4072'
    fileType = 'csv'
    window_length = 75
    emgCSV = DataProcessing(accessPath, fileType)
    emgCSV.formatCSVFiles(window_length=window_length)
    emgCSV.normalizeSet()
    classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
    classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
    
    # Paramètres à optimiser : 
    k = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
    weights_param = ['uniform', 'distance']
    weights_index = [0, 1]
    neighbors_weights_list = [(i, j) for i in weights_index for j in k]

    posture1, posture2 = 2, 4

    scoresWeightUniformPosture2 = []
    scoresWeightDistancePosture2 = []
    scoresWeightUniformPosture4 = []
    scoresWeightDistancePosture4 = []
    for config in neighbors_weights_list:
        predictions = Classifications.classifierKPlusProcheVoisins(k=config[1], weights_param=weights_param[config[0]])
        classScore, _ = Classifications.matriceDeConfusion(predictions)
        if weights_param[config[0]] == 'uniform':
            scoresWeightUniformPosture2.append(classScore[2, 2])
            scoresWeightUniformPosture4.append(classScore[4, 4])
        if weights_param[config[0]] == 'distance':
            scoresWeightDistancePosture2.append(classScore[2, 2])
            scoresWeightDistancePosture4.append(classScore[4, 4])
    
    # figure

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title('''Comparaison de la performance pour le classifier des k plus proches voisins pour la fonction de poids ''uniform'' et ''distance'' \n pour les postures 2 et 4 du dataset GEL-4072''')  # À modifier / to be modified
    ax.plot(k, scoresWeightDistancePosture2, 'r--', label="Distance Posture 2") # À compléter / to be completed
    ax.plot(k, scoresWeightUniformPosture2, 'b--', label="Uniform Posture 2")  # À compléter / to be completed
    ax.plot(k, scoresWeightDistancePosture4, 'r-', label="Distance Posture 4") # À compléter / to be completed
    ax.plot(k, scoresWeightUniformPosture4, 'b-', label="Uniform Posture 4")  # À compléter / to be completed
    ax.set_xticks(k, k)
    ax.grid(axis='x')
    ax.set_xlabel("Values of K")
    ax.set_ylabel("Accuracy [-]")
    ax.legend()

    plt.show()

optimisationClassifieurKPPV()
#do Not evaluate this current code
"""
def optimisationClassifieurAda():
    loadData()
    n_estimators=[10,50,100,200,500,1000]
    # choix des postures
    posture=[1,2,3,4,5]
    posture_bis=[1,2,3,4,5]

    posture_compare= [(i,j) for i in posture for j in posture_bis]
    posture_compare.remove((1,1))
    posture_compare.remove((2,2))
    posture_compare.remove((3,3))
    posture_compare.remove((4,4))
    posture_compare.remove((5,5))
    
    scoresAdaPosture1 = []
    scoresAdaPosture2 = []
    scoresAdaPosture3 = []
    scoresAdaPosture4 = []
    scoresAdaPosture5 = []

    for config in n_estimators:
        predictions = Classifications.ClassifieurAdaBoost(n_estimators=config, random_state=42)
        classScore, _ = Classifications.matriceDeConfusion(predictions)
        scoresAdaPosture1.append(classScore[1, 1])  
        scoresAdaPosture2.append(classScore[2, 2])
        scoresAdaPosture3.append(classScore[3, 3])
        scoresAdaPosture4.append(classScore[4, 4])
        scoresAdaPosture5.append(classScore[5, 5])
    # figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('''Comparaison de la performance pour le classifier Adaboost \n pour les 5 postures du dataset GEL-4072''')  # À modifier / to be modified
    ax.plot(n_estimators, scoresAdaPosture1, 'g--', label="Posture 1")
    ax.plot(n_estimators, scoresAdaPosture2, 'b--', label="Posture 2")  
    ax.plot(n_estimators, scoresAdaPosture3, 'o--', label="Posture 3")
    ax.plot(n_estimators, scoresAdaPosture4, 'r--', label="Posture 4") 
    ax.plot(n_estimators, scoresAdaPosture5, 'y^', label="Posture 5") 
    ax.set_xticks(n_estimators, n_estimators)
    ax.grid(axis='x')
    ax.set_xlabel("Values of estimators")
    ax.set_ylabel("Accuracy [-]")
    ax.legend()

    plt.show()

# def optimisationClassifieurDecisionTree():
#     loadData()
#     criterion = ['gini', 'entropy','log_loss']
#     max_depth = [2,4,6,8,10,12]
#     #choix des postures
#     param =dict(tree_criterion=criterion,
#                       tree_max_depth=max_depth)
#     posture=[1,2,3,4,5]
#     posture_bis=[1,2,3,4,5]

#     posture_compare= [(i,j) for i in posture for j in posture_bis]
#     posture_compare.remove((1,1))
#     posture_compare.remove((2,2))
#     posture_compare.remove((3,3))
#     posture_compare.remove((4,4))
#     posture_compare.remove((5,5))

#     for posture1,posture2 in posture: 
#         print(posture1,posture2)
       

#     scoresPosture1 = []
#     scoresPosture2 = []
#     scoresPosture3 = []
#     scoresPosture4 = []
#     scoresPosture5 = []
#     for criter in criterion :
#         for config in max_depth:
#             predictions = Classifications.ClassifieurAdaBoost(max_depth=config)
#             classScore, _ = Classifications.matriceDeConfusion(predictions)

#         scoresAdaPosture1.append(classScore[1, 1])  
#         scoresAdaPosture2.append(classScore[2, 2])
#         scoresAdaPosture3.append(classScore[3, 3])
#         scoresAdaPosture4.append(classScore[4, 4])
#         scoresAdaPosture5.append(classScore[5, 5])
#     # figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title('''Comparaison de la performance pour le classifier Adaboost \n pour les 5 postures du dataset GEL-4072''')  # À modifier / to be modified
#     ax.plot(max_depth, scoresAdaPosture1, 'g--', label="Posture 1")
#     ax.plot(max_depth, scoresAdaPosture2, 'b--', label="Posture 2")  
#     ax.plot(max_depth, scoresAdaPosture3, 'o--', label="Posture 3")
#     ax.plot(max_depth, scoresAdaPosture4, 'r--', label="Posture 4") 
#     ax.plot(max_depth, scoresAdaPosture5, 'y^', label="Posture 5") 
#     ax.set_xticks(max_depth, max_depth)
#     ax.grid(axis='x')
#     ax.set_xlabel("Values of estimators")
#     ax.set_ylabel("Accuracy [-]")
#     ax.legend()

#     plt.show()
    
# def optimisationClassifieurRandDecisionTree():
#     criterion = ['gini', 'entropy']
#     max_depth = [2,4,6,8,10,12]
"""
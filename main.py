import time
from DataProcessing import DataProcessing
from Classifications import Classifications
from Analysis import calculate_and_plot_score_vs_window
from Analysis import calculate_and_plot_score_vs_feature
## Pour l'évaluateur : 
# La grande majorité des travaux peut être observée en lançant le fichier main.py
# Le fichier DataProcessing.py va mesurer des statistiques sur les signaux EMGs et préparer les jeux de données
# Le fichier Analysis.py contient des fonctions pour analyser divers performances de nos classifieurs
# Le fichier GestureRecognitionCNN est incomplet. Le réseau de permet pas d'obtenir un score qui est représentatif d'une classification adéquate. La suite du projet va porter sur la correction de ce réseau de convolution
# Le fichier useful_functions.py contient une barre de progression qui permet de voir le temps pour certaine tâche.

# Les résultats les plus importants se retrouvent également dans le dossieur figures

# Permet de charger les données du dataset gel_4072 provenant de plusieurs utilisateurs différents (fichiers .csv) et de les traiter.
def runGel4072Dataset():
    accessPath = 'all_data/data_2_electrode_GEL_4072'
    fileType = 'csv'
    window_length = 200
    emgCSV = DataProcessing(accessPath, fileType)
    emgCSV.formatCSVFiles(window_length=window_length)
    emgCSV.normalizeSet()
    emgCSV.plotForBothDatasetsVerification(subject='0', feature='mav', k=1, ch0=6, ch1=17, classes='all')

    classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
    classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
    predictions = classifications.classifieurMethodeDeVote()
    classifications.calculate_general_score(predictions)
    classScore, classBadMeasurements = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = True)

    window_length = list(range(10, 200, 10))
    calculate_and_plot_score_vs_window(accessPath=accessPath, fileType=fileType, window_length=window_length, plot_score=True, plot_measurements=True)
    scores = calculate_and_plot_score_vs_feature(accessPath=accessPath, fileType=fileType, window_length=window_length, posture=2, plot_score=True, plot_measurements=True)

# Permet de charger les données du dataset Capgmyo provenant d'un seul utilisateur (fichiers .mat) et de les traiter.
def runCapgmyoDataset():
    accessPath = 'all_data/data_CapgMyo/matlab_format'
    fileType = 'mat'
    emgMAT = DataProcessing(accessPath, fileType)
    window_length = 900
    emgMAT.formatMATFiles(window_length=window_length, name_of_txt_file = 'data_set_all_patients_', overwrite = False)
    emgMAT.normalizeSet()
    emgMAT.plotForBothDatasetsVerification(subject='1', feature='rms', k=3, ch0=8, ch1=72, classes=[1, 2, 3, 4, 5])

    classifications = Classifications(emgMAT.emg_data, subject='1', statistique='mav', window_length=window_length)
    classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
    predictions = classifications.classifierKPlusProcheVoisins()
    classifications.calculate_general_score(predictions)
    classScore, classBadMeasurements = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = True)

    window_length = [50, 100, 150, 200, 250, 500]
    calculate_and_plot_score_vs_window(accessPath=accessPath, fileType=fileType, window_length=window_length, plot_score=True, plot_measurements=True)

start_time = time.time()
runGel4072Dataset()
runCapgmyoDataset()
end_time = time.time()

print(f'Total time = {end_time - start_time}s')

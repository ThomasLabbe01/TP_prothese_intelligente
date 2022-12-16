from DataProcessing import DataProcessing
from Classifications import Classifications
from Analysis import calculate_and_plot_score_vs_window
from Analysis import calculate_and_plot_score_vs_feature

# Permet de charger les données du dataset gel_4072 provenant de plusieurs utilisateurs différents (fichiers .csv) et de les traiter.
def runGel4072Dataset():
    accessPath = 'all_data/data_2_electrode_GEL_4072'
    fileType = 'csv'
    window_length = 25
    emgCSV = DataProcessing(accessPath, fileType)
    emgCSV.formatCSVFiles(window_length=window_length)
    emgCSV.normalizeSet()
    #emgCSV.plotForBothDatasetsVerification(subject='0', feature='mav', k=1, ch0=6, ch1=17, classes='all')

    classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
    classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
    predictions = classifications.classifieurMethodeDeVote()
    classifications.calculate_general_score(predictions)
    classScore = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = True)

# Permet de charger les données du dataset Capgmyo provenant d'un seul utilisateur (fichiers .mat) et de les traiter.
def runCapgmyoDataset():
    accessPath = 'all_data/data_CapgMyo/matlab_format'
    fileType = 'mat'
    emgMAT = DataProcessing(accessPath, fileType)
    window_length = 950
    emgMAT.formatMATFiles(window_length=950, name_of_txt_file = 'first_data_set_', overwrite = True)
    emgMAT.normalizeSet()
    #emgMAT.plotForBothDatasetsVerification(subject='1', feature='rms', k=3, ch0=8, ch1=72, classes=[1, 2, 3, 4, 5])

    #classifications = Classifications(emgMAT.emg_data, subject='1', statistique='mav', window_length=window_length)
    #classifications.data_segmentation(method='train_test_split', proportions=[0.5, 0.5, 0])
    #predictions = classifications.classifieurMethodeDeVote()
    #classifications.calculate_general_score(predictions)
    #classScore = classifications.matriceDeConfusion(predict_data = predictions, plot_figure = True)

#runGel4072Dataset()
runCapgmyoDataset()

# Run fonctions dans analysis runGel4072Dataset
accessPath = 'all_data/data_2_electrode_GEL_4072'
fileType = 'csv'
window_length = list(range(10, 200, 10))
#calculate_and_plot_score_vs_window(accessPath=accessPath, fileType=fileType, window_length=window_length, plot_figure=True)
#scores = calculate_and_plot_score_vs_feature(accessPath=accessPath, fileType=fileType, window_length=window_length, posture=2, plot_figure=True)

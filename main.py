from DataProcessing import DataProcessing
from Classifications import Classifications
from Analysis import Analysis

# Permet de charger les données du dataset gel_4072 provenant de plusieurs utilisateurs différents (fichiers .csv) et de les traiter.
def runGel4072Dataset():
    accessPath = 'all_data/data_2_electrode_GEL_4072'
    fileType = 'csv'
    emgCSV = DataProcessing(accessPath, fileType)
    emgCSV.formatCSVFiles(window_length=50)
    emgCSV.normalizeSet()
    #emgCSV.plotForBothDatasetsVerification(subject='0', feature='mav', k=1, ch0=6, ch1=17, classes='all')

    classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav')
    classifications.data_segmentation(method='train_test_split', proportions=[0.7, 0.3, 0])
    predictions = classifications.classifieurNoyauGaussien()
    classScore, classSize, classes = classifications.calculate_score_par_classe(predict_data=predictions)
    totalScore = classifications.calculate_score(classScore, classSize)
    print(totalScore)

# Permet de charger les données du dataset Capgmyo provenant d'un seul utilisateur (fichiers .mat) et de les traiter.
def runCapgmyoDataset():
    accessPath = 'all_data/data_CapgMyo/matlab_format'
    fileType = 'mat'
    emgMAT = DataProcessing(accessPath, fileType)
    emgMAT.formatMATFiles(window_length=400, name_of_txt_file = 'first_data_set_', overwrite = False)
    emgMAT.normalizeSet()
    emgMAT.plotForBothDatasetsVerification(subject='1', feature='rms', k=3, ch0=8, ch1=72, classes=[1, 2, 3, 4, 5])

runGel4072Dataset()
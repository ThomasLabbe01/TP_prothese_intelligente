import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import json
from first_draft_functions import *
# URL : http://zju-capg.org/research_en_electro_capgmyo.html
'''
The data records are in Matlab format. Each sub-database contains sss_ggg.mat for the raw data and sss_ggg_ttt.mat for the preprocessed data, 
where sss is the subject ID, ggg is the gesture ID, and ttt is the trial ID. For example, 004_001.mat contains the data (including the rest posture)
from subject 4 performing gesture 1, and 004_001_003.mat contains the preprocessed 3rd trial.
'''
'''
we present a benchmark database of HD-sEMG recordings of hand gestures performed by 23 participants, based on an 8x16 electrode array.
'''
'''
Les mouvements enregistrés contiennent donc des matrices de 1000 x 128. Avec un sampling de 1KHz, on a donc l'enregistrement de 128 électrodes pendant 1 seconde
'''
# step 1 : format emg signals to data that are easily classifiable.
# From the selected database, we have 800 movements. Every movement has a 1000 x 128 matrix associated to it. Therefore, we have 102 400 000 data point. 
# We can reduce the size of the selected data. Lets calculate the following stats for every signal : 
# 1 : Mean Absolute Value (MAV)
# 2 : Root Mean Square value (RMS)
# 3 : Variance of EMG (Var)
# 4 : Standard Deviation (SD)

# After doing so, we will have the 800 movements with 6 floats associated to every movements. So 4800 data point, of 5 different movements 
# We have 128 electrodes, 10 x 128 emg signal, each signal lasts 1s, so 1000 data 
# We have 5 movements
# We have 9 participants
# Goal : reformat data to : 
# dictionnary : {subject{data : target}}
# data : liste de liste contenant 6 dimensions (stats 1 à 6 énumérés plus haut)

#mat = scipy.io.loadmat('all_data/data_CapgMyo/test/001-001-001.mat')  # mat contient un dictionnaire dans lequel on retrouve tous les signaux des électrodes

def convert_matlab_format_to_txt_file(matlab_path):
    dict_data = {}
    list_of_data = os.listdir(matlab_path)
    for i in range(len(list_of_data)):
        subject = list_of_data[i][0:3]
        mvmnt = list_of_data[i][4:7]
        trial = list_of_data[i][8:11]
        if dict_data.get(subject) == None:
            dict_data[subject] = {'mav' : [], 'rms' : [], 'var' : [], 'sd' : [], 'target' : []} 
        emg_signals = scipy.io.loadmat(matlab_path + '/' + list_of_data[i]).get('data')
        
        dict_data[subject]['mav'].append(getMAV(emg_signals, axis=0).tolist())
        dict_data[subject]['rms'].append(getRMS(emg_signals, axis=0).tolist())
        dict_data[subject]['var'].append(getVar(emg_signals, axis=0).tolist())
        dict_data[subject]['sd'].append(getSD(emg_signals, axis=0).tolist())
        dict_data[subject]['target'].append(int(mvmnt))

    
    with open('all_data/data_CapgMyo/data_test.txt', 'w+') as data_file:
        data_file.write(json.dumps(dict_data))
    return

#path = 'all_data/data_CapgMyo/matlab_format'
#convert_matlab_format_to_txt_file(path)



data_test = json.load(open("all_data/data_CapgMyo/data_test.txt"))

# data_test[subject][datatype]
# subject : '001', '002', '003', '004', '005', '006', '007', '008', '009'
# datatype = 'mav', 'rms', 'var', 'sd', 'target'

format_test = np.array(data_test['001']['rms'])
target_test = np.array(data_test['001']['target'])
print(format_test, np.shape(format_test))  # Donc dans format_test, on retrouve 82 échantillons, et chaque échantillon possède 128 électrodes, donc 128 mesures de rms
print(target_test)
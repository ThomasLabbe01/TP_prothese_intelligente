import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from first_draft_GEL_4072 import plot_emg_and_frequency_content
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

mat = scipy.io.loadmat('all_data/data_CapgMyo/test/001-001-001.mat')  # mat contient un dictionnaire dans lequel on retrouve tous les signaux des électrodes
#np.shape(mat.get('data')) = (1000, 128)

plot_emg_and_frequency_content(mat.get('data')[0], fs=1000)

for i in range(10):
    plt.plot(mat.get('data')[i])
plt.show()


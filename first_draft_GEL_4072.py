from first_draft_functions import *
from sklearn import preprocessing

# Loading recorded EMG signals into numpy arrays
gesture_0_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-000.csv', delimiter=',')
gesture_0_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-001.csv', delimiter=',')
gesture_0_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/000-002.csv', delimiter=',')

gesture_1_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-000.csv', delimiter=',')
gesture_1_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-001.csv', delimiter=',')
gesture_1_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/001-002.csv', delimiter=',')

gesture_2_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-000.csv', delimiter=',')
gesture_2_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-001.csv', delimiter=',')
gesture_2_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-002.csv', delimiter=',')

gesture_3_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-000.csv', delimiter=',')
gesture_3_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-001.csv', delimiter=',')
gesture_3_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-002.csv', delimiter=',')

gesture_4_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-000.csv', delimiter=',')
gesture_4_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-001.csv', delimiter=',')
gesture_4_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/002-002.csv', delimiter=',')

gesture_5_0 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-000.csv', delimiter=',')
gesture_5_1 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-001.csv', delimiter=',')
gesture_5_2 = np.loadtxt('all_data/data_2_electrode_GEL_4072/003-002.csv', delimiter=',')

#print(gesture_0_0)
print(np.shape(gesture_0_0))

plot_emg_and_frequency_content(gesture_0_0[6], 1000)

path = 'all_data/data_2_electrode_GEL_4072/'

classes = [0, 1, 2, 3, 4, 5]

data, labels = segment_dataset(path, window_length=150, cha0=6, cha1=17, classes=classes)
print(labels)

features_set = features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True)
features_set = preprocessing.scale(features_set) # preprocessing module imported from sklearn


feat_x = 2  # 0-1: MAV, 2-3: RMS, 4-5: Var, 6-7: SD, 8-9: ZC, 10-11: SSC
feat_y = 3


for c in classes:
  ind = np.where(np.array(labels)==c)
  plt.scatter(features_set[ind, feat_x], features_set[ind, feat_y], label='Class '+str(c))
plt.legend()
plt.show()



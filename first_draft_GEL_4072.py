import numpy as np

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

def plot_emg_and_frequency_content():
    return
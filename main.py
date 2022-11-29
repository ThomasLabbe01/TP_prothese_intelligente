from Eletromyogram_analysis import Electromyogram_analysis
from Classifications import Classifications


# GEL_4072
path_csv = 'all_data/data_2_electrode_GEL_4072'
f_types_csv = 'csv'
emg_csv = Electromyogram_analysis(path_csv, f_types_csv)
emg_csv.format_csv_files(window_length=100)
emg_csv.verify_plots_for_both_dataset(subject='0', feature='mav', k=1, ch0=6, ch1=17, classes='all')

classifications = Classifications(emg_csv.emg_data)

exit()
# Capgmyo
path_mat = 'all_data/data_CapgMyo/matlab_format'
f_types_mat = 'mat'
emg_mat = Electromyogram_analysis(path_mat, f_types_mat)
emg_mat.format_mat_files(window_length=950, name_of_txt_file = 'first_data_set_', overwrite = False)
emg_mat.verify_plots_for_both_dataset(subject='1', feature='mav', k=3, ch0=8, ch1=72, classes=[1, 2, 3, 4, 5])


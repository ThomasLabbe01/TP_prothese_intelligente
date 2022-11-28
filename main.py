from Eletromyogram_analysis import Electromyogram_analysis


# GEL_4072
#path_csv = 'all_data/data_2_electrode_GEL_4072'
#f_types_csv = 'csv'
#emg_csv = Electromyogram_analysis(path_csv, f_types_csv)
#emg_csv.format_csv_files(window_length=300)
#data_set_csv = emg_csv.create_train_set(subject='0', feature='mav')
#emg_csv.verify_plots_for_both_dataset()  # in progress

#emg_csv.classifier_k_plus_proche_voisins(subject='0', feature='mav', k=4)
#emg_csv.plot_emg_signal_and_fft(emg_csv.emg_data['0'].get('data')[0][0])
#emg_csv.plot_hitogram_mvmnts('0')
#emg_csv.plot_parametric_classifier_2_electrodes(subject='0', ch0=6, ch1=17, classes='all', legend_with_name=False)
#emg_csv.plot_parametric_classifier(data_set_csv, classes='all', ch0=5, ch1=18)


# Capgmyo
path_mat = 'all_data/data_CapgMyo/matlab_format'
f_types_mat = 'mat'
emg_mat = Electromyogram_analysis(path_mat, f_types_mat)
emg_mat.format_mat_files(window_length=950, name_of_txt_file = 'first_data_set_', overwrite = True)
#emg_mat.normalize_set()
#data_set_mat = emg_mat.create_train_set(subject='1', feature='mav')
emg_mat.classifier_k_plus_proche_voisins(subject='4', feature='rms', k=5)
#emg_mat.verify_plots_for_both_dataset()  # in progress

#emg_mat.plot_emg_signal_and_fft(emg_mat.emg_data['1'].get('data')[0][0])
#emg_mat.plot_hitogram_mvmnts('1')
#emg_mat.plot_parametric_classifier_2_electrodes(subject=f'{1}', ch0=6, ch1=117, classes=[1, 2, 3, 4, 5, 6], legend_with_name=False)
#emg_mat.plot_parametric_classifier(data_set_mat, classes=[1, 2, 3, 4, 5], ch0=1,  ch1=72)

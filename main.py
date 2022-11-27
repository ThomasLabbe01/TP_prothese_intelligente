from Eletromyogram_analysis import Electromyogram_analysis

#path = 'all_data/data_2_electrode_GEL_4072'
#f_types = 'csv'
#test = Electromyogram_analysis(path, f_types)
#test.format_csv_files(window_length=300)
#test.plot_emg_signal_and_fft(test.emg_data['0'].get('data')[0][0])
#test.plot_hitogram_mvmnts('0')
#test.plot_jeu_2_electrodes(subject='0', ch0=6, ch1=17, classes='all', legend_with_name=False)

#data_set = test.split_data_set(subject='0', method='mav')
#test.plot_parametric_classifier(data_set, classes='all', ch0=3, ch1=18)


path = 'all_data/data_CapgMyo/matlab_format'
f_types = 'mat'
test = Electromyogram_analysis(path, f_types)
test.format_mat_files(window_length=950, name_of_txt_file = 'first_data_set_', overwrite = False)
test.normalize_set()
data_set = test.split_data_set(subject='1', method='mav')
#print(data_set[0][0])
test.plot_parametric_classifier(data_set, classes=[1, 2, 3, 4, 5], ch0=6,  ch1=81)
#test.plot_emg_signal_and_fft(test.emg_data['1'].get('data')[0][0])
#test.plot_hitogram_mvmnts('1')

#i=1
#test.plot_jeu_2_electrodes(subject=f'{i}', ch0=50, ch1=125, classes=[1, 2, 3, 4, 5, 6], legend_with_name=False)
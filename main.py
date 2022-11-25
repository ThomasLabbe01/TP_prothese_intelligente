from Eletromyogram_analysis import Electromyogram_analysis

path = 'all_data/data_2_electrode_GEL_4072'
f_types = 'csv'
test = Electromyogram_analysis(path, f_types)
test.format_csv_files(window_length=300)
#test.plot_emg_signal_and_fft(test.emg_data.get('data')[0][0])
test.plot_hitogram_mvmnts()
test.plot_jeu_2_electrodes(ch0=6, ch1=17, classes='all', legend_with_name=False)

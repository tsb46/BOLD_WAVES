import argparse
import numpy as np
import pickle

from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from run_peak_average import average_peak_window, find_comp_peaks, \
select_peaks
from preprocess.global_signal_regress import compute_global_signal
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti


def compute_gs_map(global_signal, group_data):
	global_sig_map = []
	for vertex_ts in group_data.T:
		lin_reg = LinearRegression()
		lin_reg.fit(global_signal.reshape(-1, 1), 
		            vertex_ts.reshape(-1, 1))
		global_sig_map.append(lin_reg.coef_[0][0])
	return np.array(global_sig_map)


def run_main(n_sub, input_type, n_samples=200, window=15, peak_height=2):
	# Average window around peak
	group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, False)
	group_data_gs, _, _, _ = load_data_and_stack(n_sub, input_type, True)
	global_signal = zscore(compute_global_signal(group_data))
	gs_peaks = find_comp_peaks(global_signal, peak_height)
	gs_selected_peaks = select_peaks(gs_peaks, window, window,
	                                 group_data.shape[0], n_samples)
	peak_avg = average_peak_window(gs_selected_peaks, group_data, window, window)
	peak_avg_gs = average_peak_window(gs_selected_peaks, group_data_gs, window, window)
	gs_map = compute_gs_map(global_signal, group_data)
	write_results(peak_avg, peak_avg_gs, global_signal, gs_map[np.newaxis, :], 
	              hdr, input_type, zero_mask)


def write_results(peak_avg, peak_avg_gs, global_signal, gs_map, 
                  hdr, input_type, zero_mask):
	analysis_str = f'gs'
	pickle.dump(global_signal, open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(peak_avg, hdr, 
					   peak_avg.shape[0], analysis_str+'_peak_avg')
		write_to_cifti(peak_avg_gs, hdr, 
					   peak_avg.shape[0], analysis_str+'_peak_avg_gsremoved')
		write_to_cifti(gs_map, hdr, 
					   gs_map.shape[0], analysis_str+'_map')
	elif input_type == 'gifti':
		write_to_gifti(peak_avg, hdr, analysis_str+'_peak_avg', zero_mask)
		write_to_gifti(peak_avg_gs, hdr, analysis_str+'_peak_avg_gsremoved', zero_mask)
		write_to_gifti(gs_map, hdr, analysis_str+'_map', zero_mask)


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run analytics on global signal')
	parser.add_argument('-s', '--n_sub',
						help='number of subjects to use',
						required=False,
						default=None,
						type=int)
	parser.add_argument('-t', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	args_dict = vars(parser.parse_args())
	run_main(args_dict['n_sub'], args_dict['input_type'])



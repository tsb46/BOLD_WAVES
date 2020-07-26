import argparse
import nibabel as nb 
import numpy as np
import pickle

from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from scipy.signal import find_peaks
from scipy.stats import zscore

def average_peak_window(peak_indx, group_data, window):
	windows = []
	for peak in peak_indx:
		l_edge = peak - window
		r_edge = peak + window
		windows.append(group_data[l_edge:r_edge, :])
	windows_array = np.dstack(windows)
	return np.mean(windows_array, axis=2)


def find_comp_peaks(comp_ts, sample_dist=20, height=0):
	comp_peaks = find_peaks(comp_ts, height=height, distance=sample_dist)
	return comp_peaks[0]


def run_main(input_results, n_sub, comp_number, global_signal, 
	input_type, n_samples=200, window=15):
	# Load PCA component time series and find peaks
	analysis_results = pickle.load(open(input_results, 'rb'))
	comp_ts = analysis_results['U'][:, comp_number]
	comp_peaks = find_comp_peaks(comp_ts)
	selected_peaks = select_peaks(comp_peaks, window, comp_ts.shape[0], n_samples)
	# Average window around peak
	group_data, hdr = load_data_and_stack(n_sub, input_type, global_signal)
	peak_avg = average_peak_window(selected_peaks, group_data, window)
	write_results(peak_avg, hdr, input_type, global_signal, comp_number)


def select_peaks(peaks, window, max_sample, n_samples):
	filtered_peaks = np.array([peak for peak in peaks 
	                          if peak >= window
	                          if peak <= max_sample])
	rand_peak_select = np.random.permutation(len(filtered_peaks))[:n_samples]
	return filtered_peaks[rand_peak_select]


def write_results(peak_avg, hdr, input_type, global_signal, comp_number):
	if global_signal:
		analysis_str = f'pca_peak_average_{comp_number}_gs'
	else:
		analysis_str = f'pca_peak_average_{comp_number}'
	pickle.dump(peak_avg, open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(peak_avg, hdr, 
					   peak_avg.shape[0], analysis_str)
	elif input_type == 'gifti':
		write_to_gifti(peak_avg, hdr, analysis_str)


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run peak averaging on PCA results')
	parser.add_argument('-i', '--input_results',
						help='Path to results pickle object',
						required=True,
						type=str)
	parser.add_argument('-s', '--n_sub',
						help='IMPORTANT: ensure this matches the number of '
						'you used to calculate the PCA components',
						required=True,
						type=int)
	parser.add_argument('-c', '--component',
						help='Component number',
						required=True,
						type=int)
	parser.add_argument('-g', '--gs_regress',
						help='Whether to use global signal regressed data',
						default=0,
						required=False,
						type=bool)
	parser.add_argument('-t', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	args_dict = vars(parser.parse_args())
	run_main(args_dict['input_results'], args_dict['n_sub'], 
	         args_dict['component'], args_dict['gs_regress'], 
	         args_dict['input_type'])



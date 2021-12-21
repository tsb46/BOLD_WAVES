import argparse
import nibabel as nb 
import numpy as np
import pickle

from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from scipy.signal import find_peaks
from scipy.stats import zscore

def average_peak_window(peak_indx, group_data, l_window, 
                        r_window, return_peak_ts=False):
	windows = []
	for peak in peak_indx:
		l_edge = peak - l_window
		r_edge = peak + r_window
		windows.append(group_data[l_edge:r_edge, :])
	windows_array = np.dstack(windows)
	if return_peak_ts:
		return np.mean(windows_array, axis=2), np.vstack(windows)
	else:
		return np.mean(windows_array, axis=2)


def find_comp_peaks(seed_ts, height, sample_dist=20):
	norm_ts = zscore(seed_ts)
	ts_peaks = find_peaks(norm_ts, height=height, distance=sample_dist)
	return ts_peaks[0]


def run_main(input_ts, n_sub, global_signal, input_type, 
             l_window, r_window, peak_thres, return_peak_ts, 
             n_samples=200):
	# Load time series and find peaks
	seed_ts = np.loadtxt(input_ts)
	ts_peaks = find_comp_peaks(seed_ts, peak_thres)
	selected_peaks = select_peaks(ts_peaks, l_window, r_window, 
	                              len(seed_ts), n_samples)
	# Average window around peak
	group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
	                                                    global_signal)
	if return_peak_ts:
		peak_avg, peak_ts = average_peak_window(selected_peaks, group_data, 
		                                        l_window, r_window,
		                                        return_peak_ts)
		write_results(peak_avg, hdr, input_type, global_signal, 
		              zero_mask, task, peak_ts)
	else:
		peak_avg = average_peak_window(selected_peaks, group_data, 
		                               l_window, r_window)
		write_results(peak_avg, hdr, input_type, global_signal, 
		              zero_mask)


def select_peaks(peaks, l_window, r_window, max_sample, n_samples):
	filtered_peaks = np.array([peak for peak in peaks 
	                          if peak >= l_window
	                          if (peak+r_window) <= max_sample])
	rand_peak_select = np.random.permutation(len(filtered_peaks))[:n_samples]
	return filtered_peaks[rand_peak_select]


def write_results(peak_avg, hdr, input_type, global_signal, 
                  zero_mask, peak_ts=None):
	if global_signal:
		analysis_str = 'peak_average_gs'
	else:
		analysis_str = 'peak_average_' 
	pickle.dump(peak_avg, open(f'{analysis_str}_results.pkl', 'wb'))
	if peak_ts is not None:
		pickle.dump(peak_ts, open(f'{analysis_str}_ts.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(peak_avg, hdr, 
					   peak_avg.shape[0], analysis_str)
	elif input_type == 'gifti':
		write_to_gifti(peak_avg, hdr, analysis_str, zero_mask)


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run peak averaging on '
	                                 'selected time series')
	parser.add_argument('-i', '--input_ts',
						help='Path to txt file containing seed time series',
						required=True,
						type=str)
	parser.add_argument('-s', '--n_sub',
						help='IMPORTANT: ensure this matches the number of '
						'you used to calculate seed time series',
						required=True,
						type=int)
	parser.add_argument('-g', '--gs_regress',
						help='Whether to use global signal regressed data',
						default=0,
						required=False,
						type=bool)
	parser.add_argument('-in', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	parser.add_argument('-lw', '--left_window_size',
	                    help='Length of left window from selected peak', 
	                    required=False,
	                    default=15,
	                    type=int)
	parser.add_argument('-rw', '--right_window_size',
	                    help='Length of right window from selected peak', 
	                    required=False,
	                    default=15,
	                    type=int)
	parser.add_argument('-p', '--peak_thres',
	                    help='height threshold for peak detection - set in zscore '
	                    'normalized units, i.e. std. deviations from the mean', 
	                    required=False,
	                    default=1,
	                    type=float)
	parser.add_argument('-r', '--return_peak_ts',
	                    help='Return concatenated time series within window around '
	                    'randomly selected peaks', 
	                    required=False,
	                    default=0,
	                    type=int)
	args_dict = vars(parser.parse_args())
	run_main(args_dict['input_ts'], args_dict['n_sub'], 
	         args_dict['gs_regress'], args_dict['input_type'], 
	         args_dict['left_window_size'],
	         args_dict['right_window_size'], 
	         args_dict['peak_thres'], args_dict['return_peak_ts'])



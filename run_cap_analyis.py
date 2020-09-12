import argparse
import numpy as np
import pickle

from peak_average import average_peak_window
from scipy.stats import zscore
from sklearn.cluster import KMeans
from run_seed_fc import compute_seed_ts
from sklearn.linear_model import LinearRegression
from utils.utils import load_data_and_stack, pull_gifti_data, write_to_cifti, \
write_to_gifti

"""
* Precuneus:
	LH - 1794, RH - 1384	
* Sensorimotor
	LH - 774, RH - 929
"""
def cluster_maps(selected_maps, norm_maps, n_clusters):
	if norm_maps:
		selected_maps = zscore(selected_maps.T).T
	kmeans = KMeans(n_clusters=n_clusters).fit(selected_maps)
	return kmeans.cluster_centers_, kmeans.labels_


def compute_window_average(group_data, selected_timepoints,
                           cluster_indx, n_clusters, window_size):
	# If consecutive time points are part of the same cluster, average a window
	# around the median of that point
	avg_windows_all = []
	for n in range(n_clusters):
		clus = np.where(cluster_indx == n)[0]
		ts_indx = selected_timepoints[clus]
		print('test')
		consecutive_tps = consecutive_timepoints(ts_indx)
		peaks = [np.int(np.floor(np.median(cons))) for cons in consecutive_tps]
		filtered_peaks = np.array([peak for peak in peaks 
		                          if peak >= window_size
		                          if peak <= group_data.shape[0]])
		cluster_avg = average_peak_window(filtered_peaks, group_data, window_size)
		avg_windows_all.append(cluster_avg)
	return avg_windows_all


def consecutive_timepoints(data, stepsize=1):
	# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_suprathreshold_maps(seed_signal, group_data, perc_thres):
	perc_value = np.percentile(seed_signal, perc_thres)
	selected_vals = np.where(seed_signal >= perc_value)[0]
	return selected_vals, np.squeeze(group_data[selected_vals, :]) 


def run_main(lh_vertices, rh_vertices, n_clusters, norm, n_sub, global_signal, 
             input_type, perc_thres, window_avg, window_size):
	if lh_vertices is None and rh_vertices is None:
		raise Exception('Atleast one vertex index should be supplied')

	group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
	                                                    global_signal)
	# Normalize data
	group_data = zscore(group_data)
	# Combined LH/RH data is concatenated LH then RH - add n_vertices from LH
	_, _, n_vert_L, _ = pull_gifti_data(hdr)
	# Ensure chosen vertices are not 'zeroed out' vertices - i.e. no signal
	cond_1 = all([v in zero_mask for v in lh_vertices])
	cond_2 = all([(v+n_vert_L) in zero_mask for v in rh_vertices])
	if not cond_1 or not cond_2:
		raise Exception('The vertex supplied contains all zeros')
	seed_ts = compute_seed_ts(lh_vertices, rh_vertices, group_data, zero_mask, 
	                          n_vert_L)
	selected_tps, selected_maps = get_suprathreshold_maps(seed_ts, group_data, 
	                                                      perc_thres)
	cluster_centroid, cluster_indx = cluster_maps(selected_maps, norm, n_clusters)
	if window_avg:
		cluster_win_avgs = compute_window_average(group_data, selected_tps,
		                                          cluster_indx, n_clusters, window_size)
	else:
		cluster_win_avgs = None
	write_results(cluster_centroid, cluster_indx, selected_tps,
	              cluster_win_avgs, window_avg, norm, 
	              global_signal, hdr, 
	              input_type, zero_mask)

def write_results(cluster_centroid, cluster_indx, selected_tps,
	              cluster_win_avgs, window_avg, norm, 
	              global_signal, hdr, input_type, zero_mask):
	analysis_str = 'caps'
	if global_signal:
		analysis_str += '_gs'
	if norm:
		analysis_str += '_norm'
	output = [cluster_centroid, cluster_indx, selected_tps]
	if window_avg:
		output += cluster_win_avgs
		for indx, win_avg in enumerate(cluster_win_avgs):
			tmp_str = analysis_str + f'_window_avg_clus{indx}'
			write_to_gifti(win_avg, hdr, tmp_str, zero_mask)
	pickle.dump(output, open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(cluster_centroid, hdr, 
					   cluster_centroid.shape[0], analysis_str)
	elif input_type == 'gifti':
		write_to_gifti(cluster_centroid, hdr, analysis_str, zero_mask)


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run analytics on global signal')
	parser.add_argument('-lv','--left_vertices', 
	                    default=None,
	                    action='append', 
	                    help='LH vertex indices to compute seed time series', 
	                    required=False,
	                    type=int)
	parser.add_argument('-rv','--right_vertices', 
	                    default=None,
	                    action='append', 
	                    help='RH vertex indices to compute seed time series', 
	                    required=False,
	                    type=int)
	parser.add_argument('-n', '--n_clusters',
	                    default=3, 
	                    help='Number of clusters to estimate',
	                    required=False,
	                    type=int)
	parser.add_argument('-m', '--norm_maps',
	                    default=1, 
	                    help='Whether to z-score normalize maps before clustering',
	                    required=False,
	                    type=int)
	parser.add_argument('-s', '--n_sub',
						help='number of subjects to use',
						required=False,
						default=None,
						type=int)
	parser.add_argument('-g', '--gs_regress',
						help='Whether to use global signal regressed data',
						default=0,
						required=False,
						type=int)
	parser.add_argument('-t', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	parser.add_argument('-p', '--percentile_threshold',
						help='Select the top "p" percentile BOLD values (0 through 100)',
						required=False,
						default=85,
						type=float)
	parser.add_argument('-w', '--window_average',
	                    help='Whether to average around window of selected BOLD values',
	                    required=False,
	                    default=0,
	                    type=int)
	parser.add_argument('-ws', '--window_size',
	                    help='Size of window on each side of selected BOLD value, '
	                    'i.e. a symmetric window', 
	                    required=False,
	                    default=15,
	                    type=int)
	args_dict = vars(parser.parse_args())
	run_main(args_dict['left_vertices'], args_dict['right_vertices'],
	         args_dict['n_clusters'], args_dict['norm_maps'], 
	         args_dict['n_sub'], args_dict['gs_regress'], args_dict['input_type'], 
	         args_dict['percentile_threshold'], args_dict['window_average'], 
	         args_dict['window_size'])



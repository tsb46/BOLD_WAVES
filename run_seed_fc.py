import argparse
import numpy as np
import pickle

from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from utils.utils import load_data_and_stack, pull_gifti_data, write_to_cifti, \
write_to_gifti

"""
* Precuneus:
	LH - 1794, RH - 1384	
* Sensorimotor
	LH - 774, RH - 929
*  Supramarginal Gyrus
	LH - 1371, RH - 1789
"""

def compute_fc_map(seed_signal, group_data):
	fc_map = []
	for vertex_ts in group_data.T:
		lin_reg = LinearRegression()
		lin_reg.fit(seed_signal.reshape(-1, 1), 
		            vertex_ts.reshape(-1, 1))
		fc_map.append(lin_reg.coef_[0][0])
	return np.array(fc_map)

def compute_seed_ts(lh_vertices, rh_vertices, group_data, zero_mask, n_vert_L):
	seed_ts_all = []
	if lh_vertices is not None:
		for indx in lh_vertices:
			v_indx = np.where(zero_mask==indx)
			seed_ts_all.append(group_data[:, v_indx])
	if rh_vertices is not None:
		for indx in rh_vertices:
			v_indx = np.where(zero_mask== (n_vert_L + indx))
			seed_ts_all.append(group_data[:, v_indx])
	seed_ts = np.squeeze(np.mean(seed_ts_all, axis=0))
	return seed_ts


def run_main(lh_vertices, rh_vertices, n_sub, global_signal, input_type):
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
	seed_ts = compute_seed_ts(lh_vertices, rh_vertices, group_data, zero_mask, n_vert_L)
	fc_map = compute_fc_map(seed_ts, group_data)
	write_results(seed_ts, fc_map, global_signal, hdr, input_type, zero_mask)


def write_results(seed_ts, fc_map, global_signal, hdr, input_type, zero_mask):
	analysis_str = 'fc_map'
	if global_signal:
		analysis_str += '_gs'
	pickle.dump([seed_ts, fc_map], open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(fc_map, hdr, 
					   fc_map.shape[0], analysis_str)
	elif input_type == 'gifti':
		write_to_gifti(fc_map[np.newaxis, :], hdr, analysis_str, zero_mask)


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
	parser.add_argument('-s', '--n_sub',
						help='number of subjects to use',
						required=False,
						default=None,
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
	run_main(args_dict['left_vertices'], args_dict['right_vertices'],
	         args_dict['n_sub'], args_dict['gs_regress'], 
	         args_dict['input_type'])



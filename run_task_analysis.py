import argparse
import numpy as np
import pickle

from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti

def average_task_block(blocks, group_data):
	start_ends = np.argwhere(np.diff(blocks)).squeeze()
	blocks_ts = []
	for start, end in zip(start_ends[::2], start_ends[1::2]):
		blocks_ts.append(group_data[start:(end+6), :])
	blocks_3d = np.dstack(blocks_ts)
	blocks_avg = np.mean(blocks_3d, axis=2)
	return blocks_avg


def compute_task_map(task_regressor, group_data):
	# Simple OLS - no mixed/multilevel model
	beta_map = []
	for vertex_ts in group_data.T:
		lin_reg = LinearRegression()
		lin_reg.fit(task_regressor.reshape(-1, 1), 
		            vertex_ts.reshape(-1, 1))
		beta_map.append(lin_reg.coef_[0][0])
	return np.array(beta_map)


def run_main(n_sub, input_type, task, n_samples=200, window=15):
	# Average window around peak
	group_data, hdr, \
	zero_mask, regressors = load_data_and_stack(n_sub, input_type, False, task)
	beta_map = compute_task_map(regressors[1], group_data)
	block_avg = average_task_block(regressors[0], group_data)
	write_results(block_avg, beta_map[np.newaxis, :], regressors,
	              hdr, input_type, zero_mask, task)


def write_results(block_avg, task_map, regressors,
                  hdr, input_type, zero_mask, task):
	analysis_str = 'task_' + task
	pickle.dump([regressors], open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(block_avg, hdr, 
					   block_avg.shape[0], analysis_str+'_block_avg')
		write_to_cifti(task_map, hdr, 
					   task_map.shape[0], analysis_str+'_map')
	elif input_type == 'gifti':
		write_to_gifti(block_avg, hdr, analysis_str+'_block_avg', zero_mask)
		write_to_gifti(task_map, hdr, analysis_str+'_map', zero_mask)


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run GLM and peak averaging task analysis')
	parser.add_argument('-t', '--task',
                        help='What task to apply PCA to',
                        choices=['wm', 'rel'],
                        required=True,
                        type=str)
	parser.add_argument('-s', '--n_sub',
						help='number of subjects',
						required=False,
						default=None,
						type=int)
	parser.add_argument('-i', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	args_dict = vars(parser.parse_args())
	run_main(args_dict['n_sub'], args_dict['input_type'], args_dict['task'])



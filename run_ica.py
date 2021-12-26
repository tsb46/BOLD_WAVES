import argparse
import numpy as np
import pickle

from run_main_pca import hilbert_transform
from scipy.stats import zscore
from sklearn.decomposition import FastICA
from utils.complex_fastica import complex_FastICA
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti


def run_main(n_comps, n_sub, global_signal, ica_type, 
			 real_complex, input_type):
	group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, 
														global_signal)
	# Normalize data
	if ica_type == 'spatial':
		group_data = zscore(group_data.T)
		if real_complex == 'complex':
			group_data = hilbert_transform(group_data.T).T
	elif ica_type == 'temporal':
		group_data = zscore(group_data)
		if real_complex == 'complex':
			group_data = hilbert_transform(group_data)

	# Run ICA
	if real_complex == 'real':
		unmixing_matrix, ica_comps = ica(group_data, n_comps)
	elif real_complex == 'complex':
		unmixing_matrix, _, ica_comps, _ = complex_FastICA(group_data.T, 
														   n_components = n_comps)
	if ica_type == 'spatial':
		spatial_map = ica_comps.T
		ts = unmixing_matrix
	elif ica_type == 'temporal':
		spatial_map = unmixing_matrix
		ts = ica_comps
	write_results(input_type, spatial_map, ts,
				  hdr, global_signal, zero_mask, 
				  real_complex)

def ica(input_data, n_comps):
	ica = FastICA(whiten=True, n_components=n_comps, max_iter=500)
	ica.fit(input_data)
	sources = ica.transform(input_data)
	return ica.components_, sources


def write_results(input_type, spatial_map, ica_ts,
				  hdr, global_signal, zero_mask, 
				  real_complex):
	analysis_str = 'ica'
	if global_signal:
		analysis_str += '_gs'
	if real_complex == 'complex':
		analysis_str += '_complex'
		comp_weights_real = np.real(spatial_map)
		comp_weights_imag = np.imag(spatial_map)
		comp_weights_ang = np.angle(spatial_map)

	pickle.dump([spatial_map, ica_ts], 
				open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		if real_complex == 'real':
			write_to_cifti(spatial_map, hdr, 
					   spatial_map.shape[0], analysis_str)
		elif real_complex == 'complex':
			write_to_cifti(comp_weights_real, hdr, n_comps, f'{analysis_str}_real')
			write_to_cifti(comp_weights_imag, hdr, n_comps, f'{analysis_str}_imag')
			write_to_cifti(comp_weights_ang, hdr, n_comps, f'{analysis_str}_ang')
	elif input_type == 'gifti':
		if real_complex == 'real':
			write_to_gifti(spatial_map, hdr, analysis_str, zero_mask)
		elif real_complex == 'complex':
			write_to_gifti(comp_weights_real, hdr, f'{analysis_str}_real', zero_mask)
			write_to_gifti(comp_weights_imag, hdr, f'{analysis_str}_imag', zero_mask)
			write_to_gifti(comp_weights_ang, hdr, f'{analysis_str}_ang', zero_mask)			


if __name__ == '__main__':
	"""Run main analysis"""
	parser = argparse.ArgumentParser(description='Run ICA analysis')
	parser.add_argument('-n', '--n_comps',
						help='<Required> Number of components for ICA',
						required=True,
						type=int)
	parser.add_argument('-s', '--n_sub',
						help='Number of subjects to use',
						default=None,
						type=int)
	parser.add_argument('-g', '--gs_regress',
						help='Whether to use global signal regressed data',
						default=0,
						required=False,
						type=bool)
	parser.add_argument('-type', '--ica_type',
						help='Calculate spatial or temporal ICA',
						default='spatial',
						choices=['spatial', 'temporal'], 
						type=str)
	parser.add_argument('-p', '--real_or_complex',
						help='Calculate spatial or temporal ICA',
						default='real',
						choices=['real', 'complex'], 
						type=str)
	parser.add_argument('-i', '--input_type',
						help='Whether to load resampled metric .gii files or '
						'full cifti files',
						choices=['cifti', 'gifti'],
						required=False,
						default='gifti',
						type=str)
	
	args_dict = vars(parser.parse_args())
	run_main(args_dict['n_comps'], args_dict['n_sub'], 
			 args_dict['gs_regress'],
			 args_dict['ica_type'], args_dict['real_or_complex'],
			 args_dict['input_type'])

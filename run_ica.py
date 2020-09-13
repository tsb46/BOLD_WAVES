import argparse
import numpy as np
import pickle

from scipy.stats import zscore
from sklearn.decomposition import FastICA
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti


def run_main(n_comps, n_sub, global_signal, task, ica_type, input_type):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
                                                        global_signal, 
                                                        task)
    # Normalize data
    if ica_type == 'spatial':
    	group_data = zscore(group_data.T)
    elif ica_type == 'temporal':
    	group_data = zscore(group_data)
    # Run ICA
    unmixing_matrix, ica_comps = ica(group_data, n_comps)
    if ica_type == 'spatial':
    	spatial_map = ica_comps.T
    	ts = unmixing_matrix
    elif ica_type == 'temporal':
    	spatial_map = unmixing_matrix
    	ts = ica_comps
    write_results(input_type, spatial_map, ts,
                  hdr, global_signal, zero_mask, task)

def ica(whitened_data, n_comps):
    ica = FastICA(whiten=True, n_components=n_comps, max_iter=500)
    ica.fit(whitened_data)
    sources = ica.transform(whitened_data)
    return ica.components_, sources


def write_results(input_type, spatial_map, ica_ts,
                  hdr, global_signal, zero_mask, task):
	analysis_str = 'ica'
	if global_signal:
		analysis_str += '_gs'
	pickle.dump([spatial_map, ica_ts], 
	            open(f'{analysis_str}_results.pkl', 'wb'))
	if input_type == 'cifti':
		write_to_cifti(spatial_map, hdr, 
					   spatial_map.shape[0], analysis_str)
	elif input_type == 'gifti':
		write_to_gifti(spatial_map, hdr, analysis_str, zero_mask)


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
    parser.add_argument('-t', '--task',
                        help='What task to apply PCA to',
                        choices=['rest', 'wm', 'rel'],
                        default='rest',
                        required=False,
                        type=str)
    parser.add_argument('-type', '--ica_type',
                        help='Calculate spatial or temporal ICA',
                        default='spatial',
                        choices=['spatial', 'temporal'], 
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
             args_dict['gs_regress'], args_dict['task'],
             args_dict['ica_type'], args_dict['input_type'])

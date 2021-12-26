import argparse
import numpy as np
import pickle

from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from scipy.stats import zscore
from run_main_pca import pca
from hmmlearn import hmm


def run_main(n_comps, n_sub, global_signal, input_type,
             cov_type, n_pca_comps=100):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
                                                        global_signal)
    # Normalize data
    group_data = zscore(group_data)
    # Dimension Reduction
    pca_output = pca(group_data, n_pca_comps)
    # Estimate HMM
    hmm_model, state_ts, mean_maps = gmm_hmm(pca_output, n_comps, cov_type) 
    write_results(input_type, [hmm_model, state_ts], mean_maps,
                  n_comps, hdr, global_signal, zero_mask)


def gmm_hmm(pca_output, n_comps, cov_type):
    ghmm = hmm.GMMHMM(n_components=n_comps, covariance_type=cov_type, 
                      n_iter=100)
    scores = pca_output['pc_scores']
    ghmm.fit(scores)
    pred_labels = ghmm.predict(scores)
    # Project hidden state mean vectors to original dimensions
    mean_maps = np.squeeze(ghmm.means_) @ pca_output['Va']
    return ghmm, pred_labels, mean_maps


def write_results(input_type, hmm_results, mean_maps, 
                  n_comps, hdr, global_signal, zero_mask):
    if global_signal:
        analysis_str = 'hmm_gs'
    else:
        analysis_str = 'hmm'
    pickle.dump(hmm_results, 
                open(f'{analysis_str}_results.pkl', 'wb'))
    if n_comps==1:
        mean_maps = mean_maps[np.newaxis, :]

    if input_type == 'cifti':
        write_to_cifti(mean_maps, hdr, n_comps, f'{analysis_str}_mean_map')
    elif input_type == 'gifti':
        write_to_gifti(mean_maps, hdr, f'{analysis_str}_mean_map', zero_mask)


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Estimate Gaussian Mixture HMM')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components for HMM',
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
    parser.add_argument('-i', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='gifti',
                        type=str)
    parser.add_argument('-c', '--cov_type',
                        help='Size of window for time delay embedding',
                        default='full',
                        choices=['diag', 'full', 'tied'],
                        required=False,
                        type=str)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'],
             args_dict['gs_regress'],
             args_dict['input_type'], args_dict['cov_type'])


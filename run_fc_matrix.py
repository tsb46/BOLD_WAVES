import argparse
import numpy as np
import pickle

from sklearn.decomposition import KernelPCA
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import load_data_and_stack, write_to_cifti, write_to_gifti


def compute_fc_matrix(group_data):
    # Compute FC matrix with correlation
    corr_mat = np.corrcoef(group_data.T)
    return corr_mat


def run_main(n_sub, global_signal, input_type, parcellation, dynamic_fc):
    if parcellation & (input_type == 'gifti'):
        raise Exception('Parcellation time series are saved as .ptseries.nii cifti '
                        'files - change to cifti input_type')

    group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, global_signal, parcellation)
    group_data = zscore(group_data)
    if dynamic_fc:
        # computer outer product of time point vector with itself
        fc_mat = np.apply_along_axis(outer_ltriangle, 1, group_data)
    else:
        fc_mat = compute_fc_matrix(group_data)
    write_results(fc_mat, global_signal, 'fc_matrix')


def outer_ltriangle(vec):
    outer_mat = np.outer(vec, vec.T)
    # exclude diagonal?
    vec_ltr = outer_mat[np.tril_indices(outer_mat.shape[0], k=1)]
    return vec_ltr


def write_results(fc_matrix, global_signal, analysis_str):
    if global_signal:
        analysis_str += '_gs'

    pickle.dump(fc_matrix, open(f'{analysis_str}_results.pkl', 'wb'))


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Compute Functional Connectivity Matrix')
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
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
    parser.add_argument('-p', '--parcellation', 
                        help='Whether to load parcellated time series',
                        required=False,
                        default=0,
                        type=int)
    parser.add_argument('-d', '--dynamic_fc', 
                        help='Whether to calculate dynamic FC using edge time series (Esfahlani et al. 2020)',
                        required=False,
                        default=0,
                        type=int)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_sub'], args_dict['gs_regress'],
             args_dict['input_type'], args_dict['parcellation'], 
             args_dict['dynamic_fc'])


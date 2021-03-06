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


def compute_affinity_matrix(group_data, perc_thresh):
    # Compute FC matrix with correlation
    corr_mat = np.corrcoef(group_data.T)
    # Threshold affinity mat, if specified
    if perc_thresh is not None:
        row_prct = np.percentile(corr_mat, perc_thresh, axis=1)
        mask = np.array([row < prct for row, prct in zip(corr_mat, row_prct)])
        corr_mat[mask] = 0
    # Compute affinity mat using cosine
    affinity_mat = cosine_similarity(corr_mat)
    affinity_mat[affinity_mat<0] = 0
    return affinity_mat


def kernel_pca(affinity_mat, n_comps):
    kpca = KernelPCA(n_components=n_comps, kernel='precomputed')
    kpca.fit(affinity_mat)
    embed = kpca.alphas_
    return embed.T


def spectral_embed(affinity_mat, n_comps):
    spec_emb = SpectralEmbedding(n_components=n_comps, affinity='precomputed')
    spec_emb.fit(affinity_mat)
    embed = spec_emb.embedding_
    return embed.T


def run_main(n_sub, n_comps, gradient_algorithm, 
             global_signal, input_type, perc_thresh):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, global_signal)
    group_data = zscore(group_data)
    affinity_mat = compute_affinity_matrix(group_data, perc_thresh)
    if gradient_algorithm == 'laplacian':
        embed = spectral_embed(affinity_mat, n_comps)
    else:
        embed = kernel_pca(affinity_mat, n_comps)
    write_results(embed, hdr, input_type, gradient_algorithm,
                  global_signal, zero_mask)


def write_results(emb_weights, hdr, input_type, gradient_algorithm,
                  global_signal, zero_mask):

    if gradient_algorithm == 'laplacian':
        analysis_str = 'eigenmap'
    else:
        analysis_str = 'kpca'

    if global_signal:
        analysis_str += '_gs'

    pickle.dump(emb_weights, open(f'{analysis_str}_results.pkl', 'wb'))
    if input_type == 'cifti':
        write_to_cifti(emb_weights, hdr, 
                       emb_weights.shape[0], analysis_str)
    elif input_type == 'gifti':
        write_to_gifti(emb_weights, hdr, analysis_str, zero_mask)


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run Laplacian Eigenmaps')
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-n', '--n_comps',
                        help='Number of subjects to use',
                        default=2,
                        required=False,
                        type=int)
    parser.add_argument('-a', '--gradient_algorithm',
                        help='Use Laplacian Eigenmap or PCA',
                        default='laplacian',
                        choices=['laplacian', 'pca'],
                        type=str)
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
    parser.add_argument('-p', '--percentile_threshold',
                        help='Set percentile threshold for thresholding corr matrix',
                        default=None,
                        type=float)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_sub'], args_dict['n_comps'], 
             args_dict['gradient_algorithm'], args_dict['gs_regress'],
             args_dict['input_type'], args_dict['percentile_threshold'])


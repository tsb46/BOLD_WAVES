import argparse
import fbpca
import nibabel as nb 
import numpy as np
import pickle

from mapalign.mapalign import embed
from scipy.signal import hilbert
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from utils.utils import load_data_and_stack_s3, load_data_and_stack 

# Majority of code is based on:
#https://github.com/NeuroanatomyAndConnectivity/gradient_analysis

bucket_name = 'bolt-bucket' # if you're loading from S3 bucket - 
# change to your aws S3 bucket name - make sure you configure your 
# AWS CLI for S3 access


def compute_affinity_matrix(conn_matrix):
    # Computed vectorized (flattened) affinity mat - and convert to similarity
    affinity_mat = 1 - squareform(pdist(conn_matrix, 'cosine'))
    return affinity_mat


def compute_connectivity_matrix(group_data, perc_thresh):
    corr_mat = np.corrcoef(group_data, rowvar=False)
    # Threshold any values below percentile (by row)
    row_prct = np.percentile(corr_mat, perc_thresh, axis=1)
    mask = np.array([row < prct for row, prct in zip(corr_mat, row_prct)])
    corr_mat[mask] = 0
    # Remove negative vals
    corr_mat[corr_mat < 0] = 0
    return corr_mat


def diffusion_embed(affinity_mat, alpha=0.5):
    emb = embed.compute_diffusion_map(affinity_mat, alpha=alpha)
    return emb.T


def run_main(input_dir, n_sub, perc_thresh, aws_load):
    if aws_load:
        group_data, hdr = load_data_and_stack_s3(bucket_name, n_sub)
    else:
        group_data, hdr = load_data_and_stack(input_dir, n_sub)
    conn_matrix = compute_connectivity_matrix(group_data, perc_thresh)
    # Free up memory
    del group_data
    affinity_mat = compute_affinity_matrix(conn_matrix)
    # Free up memory
    del conn_matrix
    embed = diffusion_embed(affinity_mat)
    pickle.dump(embed, open(f'diffusion_emb_n{n_comps}.pkl', 'wb'))
    write_results_to_cifti(embed, hdr)


def write_results_to_cifti(emb_weights, hdr):
    hdr_axis0  = hdr.get_axis(0)
    hdr_axis0.size = n_comps
    hdr_axis1 = hdr.get_axis(1)
    cifti_out = nb.Cifti2Image(emb_weights, (hdr_axis0, hdr_axis1))
    nb.save(cifti_out, f'diffusion_results.dtseries.nii')


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main analysis')
    parser.add_argument('-i', '--input_directory',
                        help='<Required unless loading from S3> path to '
                        'directory containing cifti files - ',
                        required=False,
                        default='',
                        type=str)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-p', '--percentile_threshold',
                        help='Set percentile threshold for thresholding corr matrix',
                        default=90,
                        type=float)
    parser.add_argument('-a', '--load_from_aws_s3',
                        help='Whether to load data from AWS S3 bucket - '
                        ' 0=No or 1=Yes',
                        default=0,
                        type=int)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_directory'], args_dict['n_sub'], 
             args_dict['percentile_threshold'], 
             args_dict['load_from_aws_s3'])


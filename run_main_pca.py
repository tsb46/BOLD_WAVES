import argparse
import fbpca
import nibabel as nb 
import numpy as np
import pickle

from glob import glob
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from scipy.signal import hilbert
from scipy.stats import zscore

bucket_name = 'bolt-bucket' # if you're loading from S3 bucket - 
# change to your aws S3 bucket name - make sure you configure your 
# AWS CLI for S3 access

def hilbert_transform(input_data):
    complex_data = hilbert(input_data, axis=0)
    return complex_data


def pca(input_data, n_comps):
    n_samples = input_data.shape[0]
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=2)
    explained_variance_ = (s ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    output_dict = {
                   'U': U,
                   's': s,
                   'Va': Va,
                   'exp_var': explained_variance_
                   }                           
    return output_dict


def run_main(n_comps, input_type, n_sub, pca_type, aws_load):
    group_data, hdr = load_data_and_stack(n_sub, input_type, 
                                          aws_load, bucket_name)
    # Normalize data
    group_data = zscore(group_data)
    # Replace NaNs w/ zeros - some vertices have no data - i.e. all 0s
    group_data[np.isnan(group_data)] = 0
    if pca_type == 'complex':
        group_data = hilbert_transform(group_data)
    elif pca_type == 'real':
        pass
    else:
        raise Exception('Only PCA types available are: "real" or "complex"')
    pca_output = pca(group_data, n_comps)
    write_results(input_type, pca_output, 
                  pca_output['Va'], n_comps, 
                  hdr, pca_type)


def write_results(input_type, pca_output, comp_weights, 
                  n_comps, hdr, pca_type):
    if pca_type == 'complex':
        pickle.dump(pca_output, open(f'pca_complex_results.pkl', 'wb'))
        comp_weights_abs = np.abs(comp_weights)
        comp_weights_ang = np.angle(comp_weights)
        if input_type == 'cifti':
            write_to_cifti(comp_weights_abs, hdr, n_comps, 'pca_complex_abs')
            write_to_cifti(comp_weights_ang, hdr, n_comps, 'pca_complex_ang')
        elif input_type == 'gifti':
            write_to_gifti(comp_weights_abs, hdr, 'pca_complex_abs')
            write_to_gifti(comp_weights_ang, hdr, 'pca_complex_ang')
    elif pca_type == 'real':
        pickle.dump(pca_output, open(f'pca_results.pkl', 'wb'))
        if input_type == 'cifti':
            write_to_cifti(comp_weights, hdr, n_comps, 'pca')
        elif input_type == 'gifti':
            write_to_gifti(comp_weights, hdr, 'pca')


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main PCA analysis')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-t', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='gifti',
                        type=str)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-p', '--pca_type',
                        help='Calculate complex or real PCA',
                        default='real',
                        type=str)
    parser.add_argument('-a', '--load_from_aws_s3',
                        help='Whether to load data from AWS S3 bucket - '
                        ' 0=No or 1=Yes',
                        default=0,
                        type=int)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'],
             args_dict['input_type'],
             args_dict['n_sub'], args_dict['pca_type'], 
             args_dict['load_from_aws_s3'])


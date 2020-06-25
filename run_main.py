import argparse
import fbpca
import nibabel as nb 
import numpy as np
import pickle

from glob import glob
from aws_utilities.load_data_from_S3_bucket import load_data_and_stack_s3
from scipy.signal import hilbert
from scipy.stats import zscore

bucket_name = 'bolt-bucket' # if you're loading from S3 bucket - 
# change to your aws S3 bucket name - make sure you configure your 
# AWS CLI for S3 access

def hilbert_transform(input_data):
    complex_data = hilbert(input_data, axis=0)
    return complex_data


def load_data_and_stack(input_dir, n_sub):
    cifti_group = []
    cifti_files = glob(input_dir + '/*dtseries.nii')
    if n_sub is None:
        n_sub = len(cifti_files)
    cifti_files_sub = cifti_files[:n_sub]
    group_data = pre_allocate_array(cifti_files_sub, n_sub)
    row_indx = 0
    for cifti_file in cifti_files_sub:
        print(cifti_file)
        cifti = nb.load(cifti_file)
        cifti.set_data_dtype('<f4')
        cifti_data = np.array(cifti.get_fdata())
        cifti.uncache() 
        n_time = cifti_data.shape[0]
        group_data[row_indx:(row_indx+n_time), :] = cifti_data
        row_indx += n_time
    hdr = cifti.header
    return group_data, hdr


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


def pre_allocate_array(cifti_files, n_sub):
    cifti = nb.load(cifti_files[0])
    n_rows, n_cols = cifti.shape
    cifti_group_arr = np.empty((n_rows*n_sub, n_cols), np.float64)

    return cifti_group_arr 


def run_main(input_dir, n_comps, n_sub, pca_type, aws_load):
    if aws_load:
        group_data, hdr = load_data_and_stack_s3(bucket_name, n_sub)
    else:
        group_data, hdr = load_data_and_stack(input_dir, n_sub)

    if pca_type == 'complex':
        group_data = hilbert_transform(group_data)
    elif pca_type == 'real':
        pass
    else:
        raise Exception('Only PCA types available are: "real" or "complex"')
    pca_output = pca(group_data, n_comps)
    pickle.dump(pca_output, open(f'pca_results_n{n_comps}.pkl', 'wb'))
    write_results_to_cifti(pca_output['Va'], n_comps, hdr, pca_type)


def write_results_to_cifti(comp_weights, n_comps, hdr, pca_type):
    hdr_axis0  = hdr.get_axis(0)
    hdr_axis0.size = n_comps
    hdr_axis1 = hdr.get_axis(1)
    if pca_type == 'complex':
        comp_weights_abs = np.abs(comp_weights)
        comp_weights_ang = np.angle(comp_weights)
        cifti_out_mag = nb.Cifti2Image(comp_weights_abs, (hdr_axis0, hdr_axis1))
        cifti_out_ang = nb.Cifti2Image(comp_weights_ang, (hdr_axis0, hdr_axis1))
        nb.save(cifti_out_mag, f'pca_complex_results_n{n_comps}_comps_abs.dtseries.nii')
        nb.save(cifti_out_ang, f'pca_complex_results_n{n_comps}_comps_ang.dtseries.nii')
    else:
        cifti_out = nb.Cifti2Image(comp_weights, (hdr_axis0, hdr_axis1))
        nb.save(cifti_out, f'pca_results_n{n_comps}_comps.dtseries.nii')


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main analysis')
    parser.add_argument('-i', '--input_directory',
                        help='<Required unless loading from S3> path to '
                        'directory containing cifti files - ',
                        required=False,
                        default='',
                        type=str)
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-t', '--pca_type',
                        help='Calculate complex or real PCA',
                        default='real',
                        type=str)
    parser.add_argument('-a', '--load_from_aws_s3',
                        help='Whether to load data from AWS S3 bucket - '
                        ' 0=No or 1=Yes',
                        default=0,
                        type=int)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_directory'], args_dict['n_comps'],
             args_dict['n_sub'], args_dict['pca_type'], 
             args_dict['load_from_aws_s3'])


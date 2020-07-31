import argparse
import fbpca
import nibabel as nb 
import numpy as np
import pickle

from pywt import swt
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from run_main_pca import pca


def multires_pca(wavelet_bank, decomp_level, n_comps):
    pca_output_grp = []
    for l in range(decomp_level):
        # Ignore approximation coefficients in first position of list
        pca_output = pca(wavelet_bank[l+1], n_comps)
        pca_output_grp.append(pca_output)
    return pca_output_grp


def run_main(input_dir, n_comps, n_sub, global_signal, 
             task_or_rest, input_type, decomp_level=5):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
                                                        global_signal, 
                                                        task_or_rest)
    # Normalize data
    group_data = zscore(group_data)
    wavelet_bank = wavelet_decomp(group_data, decomp_level)
    pca_output_grp = multires_pca(wavelet_bank, decomp_level, n_comps)
    write_results(input_type, wavelet_bank[0], 
                  pca_output_grp, n_comps, 
                  hdr, pca_type, global_signal, 
                  zero_mask)


def wavelet_decomp(group_data, decomp_level):
    wavelet_bank = swt(group_data, 'db4', decomp_level, axis=0, 
                       trim_approx=True, norm=True)
    return wavelet_bank


def write_results(input_type, wavelet_ts, pca_grps, 
                  n_comps, hdr, pca_type, global_signal,
                  zero_mask):
    import pdb; pdb.set_trace()
    if global_signal:
        analysis_str = 'pca_multires_gs'
    else:
        analysis_str = 'pca_multires'
    pickle.dump(pca_output, open(f'{analysis_str}_results.pkl', 'wb'))
    if input_type == 'cifti':
        for l, pca_output in enumerate(pca_grps):
            write_to_cifti(pca_output['Va'], hdr, n_comps, 
                           analysis_str+f'_level{l}')
    elif input_type == 'gifti':
        for l, pca_output in enumerate(pca_grps):
            write_to_gifti(pca_output['Va'], hdr, 
                           analysis_str+f'_level{l}', zero_mask)



if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main PCA analysis')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
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
    parser.add_argument('-t', '--task_or_rest',
                        help='Whether to apply to task or rest data',
                        choices=['rest', 'task'],
                        default='rest',
                        required=False,
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
             args_dict['gs_regress'], args_dict['task_or_rest'], 
             args_dict['input_type'])


import argparse
import fbpca
import numpy as np
import pickle

from numpy.linalg import pinv
from scipy.signal import hilbert
from scipy.stats import zscore
from statsmodels.tsa.vector_ar.var_model import VAR
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from utils.rotation import varimax, promax


def hilbert_transform(input_data):
    complex_data = hilbert(input_data, axis=0)
    return complex_data.conj()


def pca(input_data, n_comps, n_iter=20, l = 10):
    l += n_comps
    n_samples = input_data.shape[0]
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=n_iter, l = l)
    explained_variance_ = (s ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    pc_scores = input_data @ Va.T
    loadings =  Va.T @ np.diag(s) 
    loadings /= np.sqrt(input_data.shape[0]-1)
    output_dict = {
                   'U': U,
                   's': s,
                   'Va': Va,
                   'loadings': loadings.T,
                   'exp_var': explained_variance_,
                   'pc_scores': pc_scores
                   }   
    return output_dict


def run_main(n_comps, n_sub, global_signal, rotate, 
             input_type, pca_type, center, shuffle_ts,
             simulate_var):
    group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, global_signal)
    # Normalize data
    group_data = zscore(group_data)

    # If specified, shuffle ts (for null testing)
    if shuffle_ts:
        group_data = shuffle_time(group_data)

    # If specified, simulate null VAR model time series
    if simulate_var:
        group_data = var_simulate(group_data, group_data.shape[0])

    # If specified, center along rows
    if center == 'r':
        group_data -= group_data.mean(axis=1, keepdims=True)

    if pca_type == 'complex':
        group_data = hilbert_transform(group_data)
    pca_output = pca(group_data, n_comps)

    if rotate is not None:
        pca_output = rotation(pca_output, group_data, rotate)
    write_results(input_type, pca_output, rotate,
                  pca_output['loadings'], n_comps, 
                  hdr, pca_type, global_signal, 
                  zero_mask)


def rotation(pca_output, group_data, rotation):
    if rotation == 'varimax':
        rotated_weights, _ = varimax(pca_output['loadings'].T)
    elif rotation == 'promax':
        rotated_weights, _, _ = promax(pca_output['loadings'].T)
    # https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
    projected_scores = group_data @ pinv(rotated_weights).T
    pca_output['loadings'] = rotated_weights.T
    pca_output['pc_scores'] = projected_scores
    return pca_output


def shuffle_time(data):
    shuffle_indx = np.random.permutation(np.arange(data.shape[0]))
    return data[shuffle_indx, :]


def var_simulate(data, n_simulate, pca_n=200):
    # PCA reduction before VAR fit
    pca_dim_res = pca(data, pca_n)
    var = VAR(pca_dim_res['pc_scores'])
    var_res = var.fit(maxlags=1)
    data_sim = var_res.simulate_var(n_simulate)
    # Project simulated PCA time courses into original vertex space
    data_sim = data_sim @ pca_dim_res['Va']
    return data_sim


def write_results(input_type, pca_output, rotate, comp_weights, 
                  n_comps, hdr, pca_type, global_signal,
                  zero_mask):
    if global_signal:
        analysis_str = 'pca_gs'
    else:
        analysis_str = 'pca'
    if rotate:
        analysis_str += f'_{rotate}'
    if pca_type == 'complex':
        analysis_str += '_complex'
        pickle.dump({
                    'pca': pca_output, 
                    'metadata': [input_type, hdr, zero_mask]
                    }, open(f'{analysis_str}_results.pkl', 'wb'))
        comp_weights_real = np.real(comp_weights)
        comp_weights_imag = np.imag(comp_weights)
        comp_weights_ang = np.angle(comp_weights)
        comp_weights_amp = np.abs(comp_weights)
        if input_type == 'cifti':
            write_to_cifti(comp_weights_real, hdr, n_comps, f'{analysis_str}_real')
            write_to_cifti(comp_weights_imag, hdr, n_comps, f'{analysis_str}_imag')
            write_to_cifti(comp_weights_ang, hdr, n_comps, f'{analysis_str}_ang')
            write_to_cifti(comp_weights_amp, hdr, n_comps, f'{analysis_str}_amp')
        elif input_type == 'gifti':
            write_to_gifti(comp_weights_real, hdr, f'{analysis_str}_real', zero_mask)
            write_to_gifti(comp_weights_imag, hdr, f'{analysis_str}_imag', zero_mask)
            write_to_gifti(comp_weights_ang, hdr, f'{analysis_str}_ang', zero_mask)
            write_to_gifti(comp_weights_amp, hdr, f'{analysis_str}_amp', zero_mask)
    elif pca_type == 'real':
        pickle.dump(pca_output, open(f'{analysis_str}_results.pkl', 'wb'))
        if input_type == 'cifti':
            write_to_cifti(comp_weights, hdr, n_comps, analysis_str)
        elif input_type == 'gifti':
            write_to_gifti(comp_weights, hdr, analysis_str, zero_mask)


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
    parser.add_argument('-g', '--gs_regress',
                        help='Whether to use global signal regressed data',
                        default=0,
                        required=False,
                        type=bool)
    parser.add_argument('-r', '--rotate',
                        help='Whether to rotate pca weights',
                        default=None,
                        required=False,
                        choices=['varimax', 'promax'],
                        type=str)
    parser.add_argument('-i', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='gifti',
                        type=str)
    parser.add_argument('-p', '--real_complex',
                        help='Calculate complex or real PCA',
                        default='real',
                        choices=['real', 'complex'],
                        type=str)
    parser.add_argument('-c', '--center',
                        help='Whether to center along the columns (c) or rows (r)',
                        default='c',
                        choices=['c','r'],
                        type=str)
    parser.add_argument('-null_shuffle', '--shuffle_ts',
                        help='Whether to shuffle time series (preserving '
                             'correlation structure but removing lag)',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('-null_var', '--simulate_var',
                        help='Whether to to perform on simulated VAR(1) fit on data '
                        ' (preserving both correlation structure and (some) lag)',
                        default=0,
                        required=False,
                        type=int)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'], 
             args_dict['gs_regress'], args_dict['rotate'], 
             args_dict['input_type'], args_dict['real_complex'], 
             args_dict['center'], args_dict['shuffle_ts'], 
             args_dict['simulate_var'])


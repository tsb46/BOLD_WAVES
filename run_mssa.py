import argparse
import fbpca
import numpy as np
import pandas as pd
import pickle

from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from utils.rotation import structured_varimax
from scipy.stats import zscore
from scipy.linalg import hankel, toeplitz
from run_main_pca import pca


def run_main(n_comps, n_sub, rotate, global_signal, task_or_rest, input_type, 
             reconstruct_n, window, n_svd_comps=100):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
                                                        global_signal, 
                                                        task_or_rest)
    # Normalize data
    group_data = zscore(group_data)
    pca_output = pca(group_data, n_svd_comps)
    pc_comps = pca_output['pc_scores'].copy()
    time_delay_mat = time_delay_matrix(pc_comps, window)
    time_delay_pca = pca(time_delay_mat, n_comps)
    # Free up memory
    # del time_delay_mat
    if rotate is not None:
        time_delay_pca = rotation(time_delay_pca, n_svd_comps, window)

    time_delay_weights = restack_project_weights(time_delay_pca['Va'], window, 
                                                 n_comps, n_svd_comps, 
                                                 pca_output['U'],
                                                 group_data)
    if reconstruct_n is not None:
        recon_comps = reconstuct_components(time_delay_pca['U'], 
                                            time_delay_pca['Va'],
                                            window, reconstruct_n,
                                            n_svd_comps, 
                                            group_data.shape[0])
        # Project back onto original time series
        rc_comp = zscore(recon_comps)@np.diagflat(pca_output['s'])@pca_output['Va']
        write_to_gifti(rc_comp, hdr, f'mssa_recon_comp', 
                       zero_mask)
    write_results(input_type, time_delay_pca, 
                  time_delay_weights, n_comps, 
                  hdr, global_signal, zero_mask)


def reconstuct_components(U, Va, W, recon_n, n_ts, N):
    ts_len = U.shape[0]
    RCs = []
    RC = np.zeros((ts_len, n_ts, len(recon_n)))
    for n in recon_n:
        U_rev = time_delay_matrix(U[:,n][:, np.newaxis], W, True)   
        indx=0
        for t in range(n_ts):
            RC[:,t,n] = (U_rev @ Va[n, indx:(indx+W)])/W
            indx+= W
    return RC.sum(axis=2)


def restack_project_weights(comp_weights, window, n_comps, n_svd_comps, 
                            svd_pcs, group_data):
    component_weights = []
    for i in range(n_comps):
        restacked_weights = comp_weights[i,:].reshape((n_svd_comps, window)).T
        projected_weights = restacked_weights @ svd_pcs.T @ group_data
        component_weights.append(projected_weights)
    return component_weights


def rotation(pca_output, n_ts, window):
    rotation_mat = structured_varimax(pca_output['Va'].T, n_ts, window)
    pca_output['Va'] = (pca_output['Va'].T @ rotation_mat).T
    slen = pca_output['s'].shape[0]
    pca_output['s'] = np.diag(rotation_mat[:slen, :slen].T @ \
                              np.diag(pca_output['s']) @ \
                              rotation_mat[:slen, :slen])
    return pca_output


def time_delay_matrix(comp_ts, window, reverse=False):
    N, P = comp_ts.shape
    K =  N - window + 1
    if reverse:
        time_delay_mat = [toeplitz(comp_ts[:,p], np.zeros(window)) 
                          for p in range(P)]
    else: 
        # time_delay_mat = [hankel(comp_ts[:,p], np.zeros(window))[:K, :] 
        #                   for p in range(P)]
        time_delay_mat = [hankel(comp_ts[:,p], np.zeros(window))
                          for p in range(P)]
    return np.concatenate(time_delay_mat, axis=1)


def write_results(input_type, pca_output, comp_weights, 
                  n_comps, hdr, global_signal, zero_mask):
    if global_signal:
        analysis_str = 'mssa_gs'
    else:
        analysis_str = 'mssa'
    import pdb; pdb.set_trace()
    pickle.dump([pca_output, comp_weights], 
                open(f'{analysis_str}_results.pkl', 'wb'))
    if input_type == 'cifti':
        for i in range(n_comps):
            write_to_cifti(comp_weights[i], hdr, n_comps, 
                           analysis_str + f'_comp{i}')
    elif input_type == 'gifti':
        for i in range(n_comps):
            write_to_gifti(comp_weights[i], hdr, 
                           analysis_str + f'_comp{i}', zero_mask)


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
                        help='Whether to varimax rotate pca weights',
                        default=None,
                        required=False,
                        choices=['varimax'],
                        type=str)
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
    parser.add_argument('-recon', '--reconstruct',
                        help='Index of components to use to reconstruct time series.'
                        ' WARNING: the reconstructed output can be extremely large',
                        action='append',
                        default=None,
                        required=False,
                        type=int)
    parser.add_argument('-w', '--window',
                        help='Size of window for time delay embedding',
                        default=50,
                        required=False,
                        type=int)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'],
             args_dict['rotate'], 
             args_dict['gs_regress'], args_dict['task_or_rest'], 
             args_dict['input_type'], args_dict['reconstruct'],
             args_dict['window'])


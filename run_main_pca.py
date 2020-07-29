import argparse
import fbpca
import nibabel as nb 
import numpy as np
import pickle

from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti
from scipy.signal import hilbert
from scipy.stats import zscore
from sklearn.decomposition import MiniBatchSparsePCA


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



def run_main(n_comps, n_sub, global_signal, sparse, 
             task_or_rest, input_type, pca_type):
    group_data, hdr, zero_mask, _ = load_data_and_stack(n_sub, input_type, 
                                                        global_signal, 
                                                        task_or_rest)
    # Normalize data
    group_data = zscore(group_data)
    if pca_type == 'complex':
        group_data = hilbert_transform(group_data)
    elif pca_type == 'real':
        pass
    else:
        raise Exception('Only PCA types available are: "real" or "complex"')
    if sparse and pca_type == 'real':
        sparse_weights = sparse_pca(group_data, n_comps)
        write_to_gifti(sparse_weights, hdr, 'pca_sparse', zero_mask)
    else:
        pca_output = pca(group_data, n_comps)
        import pdb; pdb.set_trace()
        write_results(input_type, pca_output, 
                      pca_output['Va'], n_comps, 
                      hdr, pca_type, global_signal, 
                      zero_mask)


def sparse_pca(group_data, n_comps, batch_n=100, alpha=1.2):
    sparse_pca = MiniBatchSparsePCA(n_components=n_comps, 
                                    batch_size=batch_n, alpha=alpha)
    sparse_pca.fit(group_data)
    return sparse_pca.components_


def write_results(input_type, pca_output, comp_weights, 
                  n_comps, hdr, pca_type, global_signal,
                  zero_mask):
    if global_signal:
        analysis_str = 'pca_gs'
    else:
        analysis_str = 'pca'
    if pca_type == 'complex':
        analysis_str += '_complex'
        pickle.dump(pca_output, open(f'{analysis_str}_results.pkl', 'wb'))
        comp_weights_abs = np.abs(comp_weights)
        comp_weights_ang = np.angle(comp_weights)
        if input_type == 'cifti':
            write_to_cifti(comp_weights_abs, hdr, n_comps, f'{analysis_str}_abs')
            write_to_cifti(comp_weights_ang, hdr, n_comps, f'{analysis_str}_ang')
        elif input_type == 'gifti':
            write_to_gifti(comp_weights_abs, hdr, f'{analysis_str}_abs', zero_mask)
            write_to_gifti(comp_weights_ang, hdr, f'{analysis_str}_ang', zero_mask)
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
    parser.add_argument('-s', '--sparse',
                        help='Whether to use sparse PCA',
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
    parser.add_argument('-p', '--pca_type',
                        help='Calculate complex or real PCA',
                        default='real',
                        type=str)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'], 
             args_dict['gs_regress'], args_dict['sparse'], 
             args_dict['task_or_rest'], args_dict['input_type'], 
             args_dict['pca_type'])


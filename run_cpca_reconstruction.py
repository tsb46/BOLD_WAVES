import argparse
import numpy as np 
import pickle

from utils.utils import load_data_and_stack, load_gifti, pull_gifti_data, write_to_cifti, \
write_to_gifti, get_subj_file_list


def run_main(cpca_res, n_recon, n_bins, real=True):
	cpca_res = pickle.load(open(cpca_res, 'rb'))
	bin_indx_all = []
	bin_centers_all = []
	for n in range(n_recon):
		recon_ts = reconstruct_ts(cpca_res['pca'], n, real)
		phase_ts = np.angle(cpca_res['pca']['pc_scores'][:,n]) 
		bin_indx, bin_centers = create_bins(phase_ts, n_bins)
		dynamic_phase_map = create_dynamic_phase_maps(recon_ts, bin_indx, n_bins)
		bin_indx_all.append(bin_indx); bin_centers_all.append(bin_centers)
		write_to_gifti(dynamic_phase_map, cpca_res['metadata'][2], f'cpca_comp{n}_recon', 
		               cpca_res['metadata'][3])
	pickle.dump([bin_indx_all, bin_centers_all], open('cpca_reconstruction_results.pkl', 'wb'))


def create_bins(phase_ts, n_bins): 
	freq, bins = np.histogram(phase_ts, n_bins)
	bin_indx = np.digitize(phase_ts, bins)
	bin_centers = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
	return bin_indx, bin_centers


def create_dynamic_phase_maps(recon_ts, bin_indx, n_bins):
	bin_timepoints = []
	for n in range(1, n_bins+1):
		ts_indx = np.where(bin_indx==n)[0]
		bin_timepoints.append(np.mean(recon_ts[ts_indx,:], axis=0))
	dynamic_phase_map = np.array(bin_timepoints)
	return dynamic_phase_map


def reconstruct_ts(pca_res, n, real):
	U = pca_res['U'][:,n][:,np.newaxis]
	s = np.atleast_2d(pca_res['s'][n])
	Va = pca_res['Va'][n,:].conj()[np.newaxis, :]
	recon_ts = U @ s @ Va
	if real:
		recon_ts = np.real(recon_ts)
	else:
		recon_ts = np.imag(recon_ts)
	return recon_ts


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Reconstruct CPCA Components')
    parser.add_argument('-i', '--input_cpca',
                        help='<Required> File path to cpca results pickle',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_reconstruct',
                        help='<Required> Number of components to reconstruct from cPCA',
                        required=True,
                        type=int)
    parser.add_argument('-b', '--n_bins',
                        help='<Required> Number of phase bins',
                        default=30,
                        required=False,
                        type=int)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_cpca'], args_dict['n_reconstruct'], args_dict['n_bins'])
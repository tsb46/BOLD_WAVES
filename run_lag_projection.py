import argparse
import numpy as np
import pickle

from glob import glob
from numpy.linalg import pinv
from scipy.stats import zscore
from utils.utils import load_data_and_stack, write_to_cifti, \
write_to_gifti

# Majority of code is adapted from:
# https://github.com/RaichleLab/lag-code


def compute_lag_matrix(subj_data, num_nodes, tr, lags, lag_lim):
    # Normalize time series
    subj_data = zscore(subj_data)
    # Do the lagged correlation/covariance computation of TD matrices
    Cov = lagged_cov(subj_data, subj_data, np.max(lags));

    # Parabolic interpolation to get peak lag/correlation
    [pl,pc] = parabolic_interp(Cov, tr);
    pl[np.abs(pl) > lag_lim] = np.NaN # Exclude long lags (generally occur when CCF is flat)
    # Get zero-lag correlation
    temp = np.squeeze(Cov[:,:,lags==0])  # zero-lag correlation
    d = np.zeros(np.shape(temp))
    np.fill_diagonal(d, np.sqrt(np.diagonal(temp)))
    zero_lag_corr = pinv(d) @ (temp @ pinv(d))
    return pl, pc, zero_lag_corr


def compute_lag_projection(grp_lags, grp_ZL):
    # Make group-level lag projection maps
    #Unweighted lag projection
    grp_lags_proj_unweighted = np.mean(grp_lags, axis=0)
    #Weighted lag projection (inversely weight lags by correlation magnitude
    # to reduce sampling error)
    lag_weights = np.tan((np.pi/2)*(1-np.abs(grp_ZL)))**-2  #weighted by 1/f^2(r); f(r) = tan[(pi/2)(1-|r|)]
    np.fill_diagonal(lag_weights, 0)  #zero-out diagonal weights
    grp_lags_wghtd = grp_lags*lag_weights
    grp_lags_proj = np.sum(grp_lags_wghtd)/np.sum(lag_weights)
    return grp_lags_proj_unweighted, grp_lags_proj


def lagged_cov(Avg1, Avg2, L):
    L1 = Avg1.shape[1]
    L2 = Avg2.shape[1]
    corr = np.zeros((L1,L2,2*L+1), np.float32)
    k = 0
    for i in range(-L, (L+1)):
        tau = np.abs(i);
        if i == 0:
            Avg1_lagged = Avg1[0:, :]
            Avg2_lagged = Avg2[0:, :]
        elif i >0:
            Avg1_lagged = Avg1[0:-tau, :]
            Avg2_lagged = Avg2[0+tau:, :]
        else:
            Avg1_lagged = Avg1[0+tau:, :]
            Avg2_lagged = Avg2[0:-tau, :]
        
        corr[:,:,k] = Avg1_lagged.T @ Avg2_lagged
        k += 1
    return corr


def parabolic_interp(lcc, tr):
    s = lcc.shape
    peak_lag = np.squeeze(np.zeros((1,s[0]*s[1])))
    peak_cov = peak_lag.copy()
    
    # linearize
    lcc = np.reshape(lcc, [s[0]*s[1],s[2]], 'F').T
    
    #find index of extremum (max or min determined by sign at zero-lag)
    center = np.int((s[2]-1)/2)
    sign_mat = np.sign(lcc[center,:])
    I = np.argmax(lcc*sign_mat, axis=0)
    # ensure extremum is not at an endpoint (this would preclude parabolic interpolation)
    use = (I>0) & (I < (s[2]-1))
    lcc = lcc[:,use]
    # place peaks at center
    x0 = I[use] - center

    # set up three-point ccf for interpolation (y1,y2,y3)
    # i = sub2ind([size(lcc),sum(use)],I(use),1:sum(use));
    i = np.ravel_multi_index((I[use], np.arange(np.sum(use))), 
                             [lcc.shape[0], lcc.shape[1]],
                             order='F')

    lcc = np.array([lcc.ravel('F')[i-1], 
                    lcc.ravel('F')[i], 
                    lcc.ravel('F')[i+1]])
    # fit parabola: tau = TR * (y1-y3) / (2*(y1-2y2+y3))
    b = (lcc[2,:] - lcc[0,:])/2
    a = (lcc[0,:] + lcc[2,:] - 2*lcc[1,:])/2
    peak_lag[use] =  (-b/(2*a))
    # construct parabola to get covariance (y = ax^2 + bx + c)
    ax2 = a*(peak_lag[use]**2)
    bx = b*peak_lag[use]
    c = lcc[1,:]
    peak_cov[use]= ax2 + bx + c

    # put back TR information
    peak_lag[use] = (peak_lag[use] + x0)*tr

    peak_lag = np.reshape(peak_lag,[s[0], s[1]], 'F')
    peak_cov = np.reshape(peak_cov,[s[0], s[1]], 'F')

    return peak_lag, peak_cov


def run_lag_projection(input_data, tr=0.72, lag=8, lag_lim=12):
   # lag limit (in seconds)
    lags = np.arange(-lag,(lag+1))  # range of TR shifts; max(lags) = round(lag_lim/tr + 1)
    num_nodes = input_data.shape[1]
    lag_mat, cov_mat, zero_lag_corr = compute_lag_matrix(input_data, 
                                                         num_nodes, 
                                                         tr, lags,
                                                         lag_lim)
    uw_lag_proj, w_lag_proj = compute_lag_projection(lag_mat,
                                                     zero_lag_corr)
    return uw_lag_proj, w_lag_proj, cov_mat

def run_main(input_type, global_signal, n_sub):
    group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, global_signal)
    lag_results = run_lag_projection(group_data)
    write_results(input_type, lag_results, 
                  lag_results[0][np.newaxis, :], hdr,
                  global_signal, zero_mask)


def write_results(input_type, lag_results, lag_projection, hdr, 
                  global_signal, zero_mask):
    if global_signal:
        analysis_str = 'lag_projection_gs'
    else:
        analysis_str = 'lag_projection'
    pickle.dump(lag_results, 
            open(f'{analysis_str}_results.pkl', 'wb'))
    if input_type == 'cifti':
        write_to_cifti(lag_projection, hdr, n_comps, analysis_str)
    elif input_type == 'gifti':
        write_to_gifti(lag_projection, hdr, analysis_str, zero_mask)


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run lag projection analysis')
    parser.add_argument('-t', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='gifti',
                        type=str)
    parser.add_argument('-g', '--gs_regress',
                        help='Whether to use global signal regressed data',
                        default=0,
                        required=False,
                        type=bool)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_type'], args_dict['gs_regress'],
             args_dict['n_sub'])


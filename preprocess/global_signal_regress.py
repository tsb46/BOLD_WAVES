import argparse
import nibabel as nb
import numpy as np
import os
import sys

# Fix sub-module import - messing with system path not ideal
sys.path.append(os.getcwd())

from sklearn.linear_model import LinearRegression
from utils.utils import load_gifti, pull_gifti_data, write_to_gifti

def compute_global_signal(subj_data):
    return subj_data.mean(axis=1)


def regress_global_signal(subj_data, global_signal):
    residual_ts_all = []
    global_sig_map = []
    for vertex_ts in subj_data.T:
        lin_reg = LinearRegression()
        lin_reg.fit(global_signal.reshape(-1, 1), 
                    vertex_ts.reshape(-1, 1))
        global_sig_map.append(lin_reg.coef_[0][0])
        pred_ts = lin_reg.predict(global_signal.reshape(-1,1))
        residual_ts = vertex_ts.reshape(-1,1) - pred_ts
        residual_ts_all.append(residual_ts)
    return np.squeeze(np.array(residual_ts_all)).T, np.array(global_sig_map)


def run_main(gifti_files, output_dir):
    # Create global signal map dir in output_dir
    if not os.path.isdir(output_dir + '/gs_maps'):
        os.mkdir(output_dir + '/gs_maps')
    if not os.path.isdir(output_dir + '/gs_ts'):
        os.mkdir(output_dir + '/gs_ts')
    gifti = load_gifti(gifti_files)
    subj_data, n_time, _, _ = pull_gifti_data(gifti)
    hdr = gifti
    zero_mask = np.std(subj_data, axis=0) > 0
    zero_mask_indx = np.where(zero_mask)[0]
    subj_data = subj_data[:, zero_mask_indx]
    global_signal = compute_global_signal(subj_data)
    subj_data_residuals, global_sig_map = regress_global_signal(subj_data, 
                                                                global_signal)
    write_data(gifti_files, global_signal, subj_data_residuals, 
               global_sig_map, output_dir, hdr, zero_mask_indx) 


def write_data(gifti_files, global_signal, residuals, 
               global_sig_map, output_dir, hdr, zero_mask):
    base_file_dt = os.path.splitext(os.path.basename(gifti_files[0]))[0]
    base_file2 = os.path.splitext(base_file_dt)[0]
    base_file = os.path.splitext(base_file2)[0]
    # Write output
    np.savetxt(output_dir + '/gs_ts/' + base_file + '_gs_ts.txt', global_signal)
    output_subj_data = output_dir + '/' + base_file + '_filt_gs'
    write_to_gifti(residuals, hdr, output_subj_data, zero_mask, cifti=False)
    output_gs_map = output_dir + '/gs_maps/' + base_file + '_gs_map'
    write_to_gifti(global_sig_map[np.newaxis, :], hdr, output_gs_map, 
                   zero_mask, cifti=False)


if __name__ == '__main__':
    """Bandpass filter w/ Butterworth Filter """
    parser = argparse.ArgumentParser(description='Regress out global signal')
    parser.add_argument('-l', '--gifti_left',
                        help='<Required> path to gifti file - left hemisphere',
                        required=True,
                        type=str)
    parser.add_argument('-r', '--gifti_right',
                        help='<Required> path to gifti file - right hemisphere',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='the repetition time of the data',
                        required=False,
                        default=os.getcwd(),
                        type=str)
    args_dict = vars(parser.parse_args())
    run_main([args_dict['gifti_left'], args_dict['gifti_right']],
             args_dict['output_dir'])

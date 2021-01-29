import argparse
import nibabel as nb
import numpy as np
import os

from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.stats import zscore

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # Use filtfilt to avoid phase delay
    data_filt = sosfiltfilt(sos, data, axis=0)
    return data_filt


def output_cifti(cifti_file, cifti, proc_data, output_dir):
    base_file_dt = os.path.splitext(os.path.basename(cifti_file))[0]
    base_file = os.path.splitext(base_file_dt)[0]
    cifti_out = nb.Cifti2Image(proc_data, cifti.header)
    cifti_out.set_data_dtype('<f4')
    nb.save(cifti_out, output_dir + '/' + base_file + '_filt.dtseries.nii')


def run_main(cifti_file, low, high, tr, output_dir):
    cifti = nb.load(cifti_file)
    cifti_data = np.array(cifti.get_fdata())
    cifti.uncache()
    # Get sampling rate
    fs = 1 / tr
    # Demean data
    cifti_data = zscore(cifti_data)
    # Bandpass filter with Butterworth
    data_filt = butter_bandpass_filter(cifti_data, low, high, fs)
    # Write output
    output_cifti(cifti_file, cifti, data_filt, output_dir)


if __name__ == '__main__':
    """Bandpass filter w/ Butterworth Filter """
    parser = argparse.ArgumentParser(description='Normalize and bandpass filter cifti files')
    parser.add_argument('-c', '--cifti',
                        help='<Required> path to cifti file',
                        required=True,
                        type=str)
    parser.add_argument('-l', '--low_cut',
                        help='<Required> Lower bound of the bandpass filter',
                        required=True,
                        type=float)
    parser.add_argument('-u', '--high_cut',
                        help='<Required> Higher bound of the bandpass filter',
                        required=True,
                        type=float)
    parser.add_argument('-t', '--tr',
                        help='the repetition time of the data',
                        required=False,
                        default=0.72,
                        type=float)
    parser.add_argument('-o', '--output_dir',
                        help='the repetition time of the data',
                        required=False,
                        default=os.getcwd(),
                        type=str)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['cifti'], args_dict['low_cut'],
             args_dict['high_cut'], args_dict['tr'],
             args_dict['output_dir'])

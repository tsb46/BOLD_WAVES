import argparse
import fbpca
import nibabel as nb 
import numpy as np
import os
import pickle

from numpy import ndarray
from numpy.random import RandomState
from scipy.signal import find_peaks
from utils.utils import load_data_and_stack_s3, load_data_and_stack 

# Code is a reconfiguration of:
# https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/qpp/qpp.py

bucket_name = 'bolt-bucket' # if you're loading from S3 bucket - 
# change to your aws S3 bucket name - make sure you configure your 
# AWS CLI for S3 access


def correlation_threshold(high_thresh, low_thresh, thresh_iter):
    """
    Use the threshold procedure described in 
    Behnaze et al. 2018 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5845807/
    First three iterations is lower threshold, rest is higher threshold
    """
    if thresh_iter < 4:
        return [low_thresh]*(thresh_iter-1) + [high_thresh]
    else:
        return [low_thresh] * 3 + [high_thresh] * (thresh_iter-3)

def smooth(x):
    """
    Temporary moving average
    """
    return np.array(
        [x[0]] +
        [np.mean(x[0:3])] +
        (np.convolve(x, np.ones(5), 'valid') / 5).tolist() +
        [np.mean(x[-3:])] +
        [x[-1]]
    )


def flattened_segment(data, window_length, pos):
    return data[:, pos:pos + window_length].flatten(order='F')


def normalize_segment(segment, df):
    segment -= np.sum(segment) / df
    segment = segment / np.sqrt(np.dot(segment, segment))
    return segment


def detect_qpp(data, window_length, num_scans,
               permutations=10, low_corr=0.1,
               high_corr=0.3, thresh_iter=20,
               convergence_iterations=1,
               random_state=None):
    """
    FROM https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/qpp/qpp.py:
    This code is adapted from the paper "Quasi-periodic patterns (QP): Large-
    scale dynamics in resting state fMRI that correlate with local infraslow
    electrical activity", Shella Keilholz et al. NeuroImage, 2014.
    """

    random_state = RandomState(random_state)

    voxels, trs = data.shape

    iterations = int(max(1, thresh_iter))
    convergence_iterations = int(max(1, convergence_iterations))

    correlation_thresholds = correlation_threshold(high_corr, 
                                                   low_corr, 
                                                   iterations)
    trs_per_scan = int(trs / num_scans)
    inpectable_trs = np.arange(trs) % trs_per_scan
    inpectable_trs = np.where(inpectable_trs < trs_per_scan - window_length + 1)[0]

    df = voxels * window_length

    initial_trs = random_state.choice(inpectable_trs, permutations)

    permutation_result = [{} for _ in range(permutations)]
    for perm in range(permutations):

        template_holder = np.zeros(trs)
        random_initial_window = normalize_segment(flattened_segment(data, window_length, initial_trs[perm]), df)
        for tr in inpectable_trs:
            scan_window = normalize_segment(flattened_segment(data, window_length, tr), df)
            template_holder[tr] = np.dot(random_initial_window, scan_window)

        template_holder_convergence = np.zeros((convergence_iterations, trs))

        for iteration in range(iterations):
            print(iteration)
            peak_threshold = correlation_thresholds[iteration]

            peaks, _ = find_peaks(template_holder, height=peak_threshold, distance=window_length)
            peaks = np.delete(peaks, np.where(~np.isin(peaks, inpectable_trs))[0])

            template_holder = smooth(template_holder)

            found_peaks = np.size(peaks)
            if found_peaks < 1:
                break

            peaks_segments = flattened_segment(data, window_length, peaks[0])
            for peak in peaks[1:]:
                peaks_segments = peaks_segments + flattened_segment(data, window_length, peak)

            peaks_segments = peaks_segments / found_peaks
            peaks_segments = normalize_segment(peaks_segments, df)

            for tr in inpectable_trs:
                scan_window = normalize_segment(flattened_segment(data, window_length, tr), df)
                template_holder[tr] = np.dot(peaks_segments, scan_window)

            if np.all(np.corrcoef(template_holder, template_holder_convergence) > 0.9999):
                break

            if convergence_iterations > 1:
                template_holder_convergence[1:] = template_holder_convergence[0:-1]
            template_holder_convergence[0] = template_holder

        if found_peaks > 1:
            permutation_result[perm] = {
                'template': template_holder,
                'peaks': peaks,
                'final_iteration': iteration,
                'correlation_score': np.sum(template_holder[peaks]),
            }

    # Retrieve max correlation of template from permutations
    correlation_scores = np.array([
        r['correlation_score'] if r else 0.0 for r in permutation_result
    ])
    if not np.any(correlation_scores):
        raise Exception("C-PAC could not find QPP in your data. "
                        "Please lower your correlation threshold and try again.")

    max_correlation = np.argsort(correlation_scores)[-1]
    best_template = permutation_result[max_correlation]['template']
    best_selected_peaks = permutation_result[max_correlation]['peaks']

    best_template_metrics = [
        np.median(best_template[best_selected_peaks]),
        np.median(np.diff(best_selected_peaks)),
        len(best_selected_peaks),
    ]

    window_length_start = round(window_length / 2)
    window_length_end = window_length_start - window_length % 2

    best_template_segment = np.zeros((voxels, window_length))

    for best_peak in best_selected_peaks:
        start_tr = int(best_peak - np.ceil(window_length / 2.))
        end_tr = int(best_peak + np.floor(window_length / 2.))

        start_segment = np.zeros((voxels, 0))
        if start_tr <= 0:
            start_segment = np.zeros((voxels, abs(start_tr)))
            start_tr = 0

        end_segment = np.zeros((voxels, 0))
        if end_tr > trs:
            end_segment = np.zeros((voxels, end_tr - trs))
            end_tr = trs

        data_segment = data[:, start_tr:end_tr]

        best_template_segment += np.concatenate([
            start_segment,
            data_segment,
            end_segment,
        ], axis=1)

    best_template_segment /= len(best_selected_peaks)
    packaged_results = [best_template_segment, best_selected_peaks, 
                        best_template_metrics, best_template]
    return packaged_results


def run_main(input_dir, n_sub, window_length, aws_load):
    if aws_load:
        group_data, hdr = load_data_and_stack_s3(bucket_name, n_sub)
    else:
        group_data, hdr = load_data_and_stack(input_dir, n_sub)
    qpp_results = detect_qpp(group_data.T, 
                                                      window_length, 
                                                      n_sub)
    pickle.dump(qpp_results, 
                open(f'qpp_results.pkl', 'wb'))
    write_results_to_cifti(qpp_results[0].T, hdr)


def write_results_to_cifti(segment, hdr):
    hdr_axis0  = hdr.get_axis(0)
    hdr_axis0.size = segment.shape[0]
    hdr_axis1 = hdr.get_axis(1)
    cifti_out = nb.Cifti2Image(segment, (hdr_axis0, hdr_axis1))
    nb.save(cifti_out, f'qpp_results.dtseries.nii')


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main analysis')
    parser.add_argument('-i', '--input_directory',
                        help='<Required unless loading from S3> path to '
                        'directory containing cifti files - ',
                        required=False,
                        default='',
                        type=str)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-w', '--window_length',
                        help='Set window length for QPP',
                        default=30,
                        type=int)
    parser.add_argument('-a', '--load_from_aws_s3',
                        help='Whether to load data from AWS S3 bucket - '
                        ' 0=No or 1=Yes',
                        default=0,
                        type=int)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_directory'], args_dict['n_sub'], 
             args_dict['window_length'], 
             args_dict['load_from_aws_s3'])

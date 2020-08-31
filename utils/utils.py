import matplotlib.pyplot as plt
import nibabel as nb 
import numpy as np
import os

from glob import glob
from itertools import groupby 
from matplotlib.patches import Rectangle
from nibabel import FileHolder, Cifti2Image, GiftiImage 
from nibabel.gifti.gifti import GiftiDataArray
from scipy.stats import gamma, zscore


def assign_gifti_files_to_nested_list(gifti_files):
	id_extract = lambda x: x.split('/')[-1].split('_')[0]
	temp = sorted(gifti_files, key = id_extract) 
	nested_gifti_list = [list(subj_id) for i, subj_id in groupby(temp, id_extract)] 
	return nested_gifti_list
	

def construct_task_blocks(scan_len, timings, durs, tr):
	task_blocks = np.zeros(scan_len)
	for event, dur in zip(timings, durs):
		tr_event = np.int(np.floor(event/tr) - 1)
		tr_dur = np.int(np.floor(dur))
		task_blocks[tr_event:(tr_event+tr_dur)] = 1
	return task_blocks


def construct_task_regressors(scan_len, n_subs, task, hrf_t=30, tr=0.72):
	# We're modeling all task events as the same event,  
	# in this case, the task timing of LR and RL encoding scans are the same
	if task == 'wm':
		LR_evs_fps = sorted(glob('data/task_LR_ev/wm/*.txt'))
		LR_evs = [np.loadtxt(fp) for fp in LR_evs_fps]
		LR_start = [LR_ev[0] for LR_ev in LR_evs]
		LR_dur = [LR_ev[1] for LR_ev in LR_evs]
	elif task == 'rel':
		LR_evs_fps = sorted(glob('data/task_LR_ev/rel/*.txt'))
		LR_evs = [np.loadtxt(fp) for fp in LR_evs_fps]
		LR_start = [LR_ev[0] for LR_ev in LR_evs[0]] + \
		[LR_ev[0] for LR_ev in LR_evs[1]] 
		LR_dur = [LR_ev[1] for LR_ev in LR_evs[0]] + \
		[LR_ev[1] for LR_ev in LR_evs[1]] 
	LR_blocks = construct_task_blocks(scan_len, LR_start, LR_dur, tr)
	hrf = double_gamma_hrf(hrf_t, tr)
	LR_convolved = convolve_hrf_events(hrf, LR_blocks)
	return np.tile(LR_blocks, n_subs), np.tile(LR_convolved, n_subs)


def convolve_hrf_events(hrf, blocks):
	n_drop = len(hrf) - 1
	convolved_events = np.convolve(blocks, hrf)
	return convolved_events[:-n_drop]


def double_gamma_hrf(t, tr):
	# http://www.jarrodmillman.com/rcsds/lectures/convolution_background.html
	n_steps = np.arange(0, t, tr)
	gamma_peak = gamma.pdf(n_steps, 6)
	gamma_under = gamma.pdf(n_steps, 12)
	gamma_double = gamma_peak - 0.35 * gamma_under
	return gamma_double / np.max(gamma_double) * 0.6


def get_subj_file_list(n_sub, input_type, global_signal, 
                       task, multi_res):
	if input_type == 'cifti':
		if global_signal:
			raise Exception('Global signal regression was '
			                'only conducted for gifti processed files')
		if task == 'rest':
			subj_files = sorted(glob('data/rest/proc_2_norm_filter/*dtseries.nii'))
		else:
			if task == 'wm':
				subj_files = sorted(glob('data/task/wm/proc_2_norm_filter/*dtseries.nii'))
			elif task == 'rel':
				subj_files = sorted(glob('data/task/rel/proc_2_norm_filter/*dtseries.nii'))
	elif input_type == 'gifti':
		if global_signal:
			if task == 'rest':
				subj_files = glob('data/rest/proc_4_gs_regress/*.gii')
			else:
				raise Exception('Global signal regression was not performed for task data')
		else:
			if task == 'rest':
				if multi_res:
					subj_files = glob('data/rest/proc_2_surf_resamp/*.gii')
				else:
					subj_files = glob('data/rest/proc_3_surf_resamp/*.gii')
			else:
				if task == 'wm':
					subj_files = glob('data/task/wm/proc_3_surf_resamp/*.gii')
				elif task == 'rel':
					subj_files = glob('data/task/rel/proc_3_surf_resamp/*.gii')
		subj_files = assign_gifti_files_to_nested_list(subj_files)
	if len(subj_files) < 1:
		raise Exception('No files found in file path')
	if n_sub is None:
		n_sub = len(subj_files)
	subj_files_sub = subj_files[:n_sub]
	return subj_files_sub, n_sub


def load_cifti(cifti_fp):
	cifti = nb.load(cifti_fp)
	return cifti


def load_gifti(gifti_fps):
	gifti_LR = []
	file_base = gifti_fps[0].split('.')[0]
	for hem in ['L', 'R']:
		indx = gifti_fps.index(f'{file_base}.{hem}.func.gii')
		gifti_LR.append(nb.load(gifti_fps[indx]))
	return gifti_LR


def load_data_and_stack(n_sub, input_type, global_signal, 
                        task='rest', multi_res=False):
	subj_files, n_sub = get_subj_file_list(n_sub, input_type, global_signal, 
	                                       task, multi_res)
	group_data, n_rows = pre_allocate_array(subj_files[0], input_type, n_sub)
	row_indx = 0
	for subj_file in subj_files:
		print(subj_file)
		if input_type == 'cifti':
			subj_file = load_cifti(subj_file)
			hdr, subj_data, n_time = pull_cifti_data(subj_file)
		elif input_type == 'gifti':
			subj_file = load_gifti(subj_file)
			subj_data, n_time, _, _ = pull_gifti_data(subj_file)
			if multi_res:
				# Multi-resolution data has not been previously normalized - do this
				# before concatenation
				subj_data = zscore(subj_data)
			hdr = subj_file
		group_data[row_indx:(row_indx+n_time), :] = subj_data
		row_indx += n_time
	# Construct mask of vertices with all 0s
	zero_mask = np.std(group_data, axis=0) > 0
	zero_mask_indx = np.where(zero_mask)[0]
	if task != 'rest':
		regressor = construct_task_regressors(n_rows, n_sub, task)
	else:
		regressor = None
	return group_data[:, zero_mask], hdr, zero_mask_indx, regressor


def plot_sorted_corr_mat(corr_mat, cluster_assignments):
	# Create sorting index from factor assignments
	sort_indx = np.argsort(factor_assignments)
	sorted_vals = np.sort(factor_assignments)
	# Sort Distance Matrix
	sortedmat = [[dist_mat[i][j] for j in sort_indx] for i in sort_indx]
	# Plot Distance Matrix
	fig, ax = plt.subplots(figsize=(10,7))
	c = plt.pcolormesh(sortedmat, cmap='seismic')
	fig.colorbar(c, ax=ax)
	# Plot rectangular patches along diagnol to indicate factor assignments
	for i in np.unique(sorted_vals):
		ind = np.where(sorted_vals == i)
		mn = np.min(ind)
		mx = np.max(ind)
		sz=(mx-mn)+1
		rect = Rectangle((mn,mn), sz, sz ,linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
	plt.show()

def pull_cifti_data(cifti_obj):
	cifti_obj.set_data_dtype('<f4')
	cifti_data = np.array(cifti_obj.get_fdata())
	cifti_obj.uncache()
	n_time = cifti_data.shape[0]
	return cifti_obj.header, cifti_data, n_time


def pull_gifti_data(giftis):
	# LH and RH data are concatenated with LH first and RH second!
	gifti_L_array = np.array(giftis[0].agg_data())
	gifti_R_array = np.array(giftis[1].agg_data())
	gifti_all = np.concatenate((gifti_L_array, gifti_R_array), axis=1)
	n_time = gifti_all.shape[0]
	n_vert_L = gifti_L_array.shape[1]
	n_vert_R = gifti_R_array.shape[1]
	return gifti_all, n_time, n_vert_L, n_vert_R


def pre_allocate_array(subj_file, input_type, n_sub):
	if input_type == 'cifti':
		subj = load_cifti(subj_file)	
	elif input_type == 'gifti':
		subj = load_gifti(subj_file)
		subj, _, _, _ = pull_gifti_data(subj)
	n_rows, n_cols = subj.shape
	group_array = np.empty((n_rows*n_sub, n_cols), np.float64)
	return group_array, n_rows


def write_to_cifti(result, hdr, n_rows, script_name):
	hdr_axis0  = hdr.get_axis(0)
	hdr_axis0.size = n_rows
	hdr_axis1 = hdr.get_axis(1)
	cifti_out = nb.Cifti2Image(result, (hdr_axis0, hdr_axis1))
	nb.save(cifti_out, f'{script_name}_results.dtseries.nii')


def write_to_gifti(result, giftis, script_name, zero_mask):
	example_array_L = giftis[0].darrays[0]
	example_array_R = giftis[1].darrays[0]		
	L_shape = len(giftis[0].agg_data()[0])
	padded_result = np.zeros([result.shape[0], L_shape*2])
	padded_result[:, zero_mask] = result
	L_result = padded_result[:, :L_shape]
	R_result = padded_result[:, L_shape:]
	L_gifti_image = GiftiImage(meta=giftis[0].meta)
	R_gifti_image = GiftiImage(meta=giftis[1].meta)
	for row_L, row_R in zip(L_result, R_result):
		gifti_array_L = GiftiDataArray(row_L, intent=example_array_L.intent,
			datatype=example_array_L.datatype, meta=example_array_L.meta)
		gifti_array_R = GiftiDataArray(row_R, intent=example_array_R.intent,
			datatype=example_array_R.datatype, meta=example_array_R.meta)
		L_gifti_image.add_gifti_data_array(gifti_array_L)
		R_gifti_image.add_gifti_data_array(gifti_array_R)
	nb.save(L_gifti_image, f'{script_name}.L.func.gii')
	nb.save(R_gifti_image, f'{script_name}.R.func.gii')
	os.system(f'./utils/giftis_to_cifti.sh {script_name}.L.func.gii '
	          f'{script_name}.R.func.gii {script_name}.dtseries.nii')













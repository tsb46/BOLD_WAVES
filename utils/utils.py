import nibabel as nb 
import numpy as np
import os

from glob import glob
from itertools import groupby 
from nibabel import FileHolder, Cifti2Image, GiftiImage 
from nibabel.gifti.gifti import GiftiDataArray


def assign_gifti_files_to_nested_list(gifti_files):
	id_extract = lambda x: x.split('/')[-1].split('_')[0]
	temp = sorted(gifti_files, key = id_extract) 
	nested_gifti_list = [list(subj_id) for i, subj_id in groupby(temp, id_extract)] 
	return nested_gifti_list


def get_subj_file_list(n_sub, input_type, global_signal, task_or_rest):
	if input_type == 'cifti':
		if global_signal:
			raise Exception('Global signal regression was '
			                'only conducted for gifti processed files')
		if task_or_rest == 'rest':
			subj_files = sorted(glob('data/rest/proc_2_norm_filter/*dtseries.nii'))
		else:
			subj_files = sorted(glob('data/task/proc_2_norm_filter/*dtseries.nii'))
	elif input_type == 'gifti':
		if global_signal:
			if task_or_rest == 'rest':
				subj_files = glob('data/rest/proc_4_gs_regress/*.gii')
			else:
				subj_files = glob('data/task/proc_4_gs_regress/*.gii')
		else:
			if task_or_rest == 'rest':
				subj_files = glob('data/rest/proc_3_surf_resamp/*.gii')
			else:
				subj_files = glob('data/task/proc_3_surf_resamp/*.gii')
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


def load_data_and_stack(n_sub, input_type, global_signal, task_or_rest='rest'):
	subj_files, n_sub = get_subj_file_list(n_sub, input_type, global_signal, task_or_rest)
	group_data = pre_allocate_array(subj_files[0], input_type, n_sub)
	row_indx = 0
	for subj_file in subj_files:
		print(subj_file)
		if input_type == 'cifti':
			subj_file = load_cifti(subj_file)
			hdr, subj_data, n_time = pull_cifti_data(subj_file)
		elif input_type == 'gifti':
			subj_file = load_gifti(subj_file)
			subj_data, n_time = pull_gifti_data(subj_file)
			hdr = subj_file
		group_data[row_indx:(row_indx+n_time), :] = subj_data
		row_indx += n_time
	zero_mask = np.std(group_data, axis=0) > 0
	zero_mask_indx = np.where(zero_mask)[0]
	return group_data[:, zero_mask], hdr, zero_mask_indx 


def pull_cifti_data(cifti_obj):
	cifti_obj.set_data_dtype('<f4')
	cifti_data = np.array(cifti_obj.get_fdata())
	cifti_obj.uncache()
	n_time = cifti_data.shape[0]
	return cifti_obj.header, cifti_data, n_time


def pull_gifti_data(giftis):
	gifti_L_array = np.array(giftis[0].agg_data())
	gifti_R_array = np.array(giftis[1].agg_data())
	gifti_all = np.concatenate((gifti_L_array, gifti_R_array), axis=1)
	n_time = gifti_all.shape[0]
	return gifti_all, n_time


def pre_allocate_array(subj_file, input_type, n_sub):
	if input_type == 'cifti':
		subj = load_cifti(subj_file)	
	elif input_type == 'gifti':
		subj = load_gifti(subj_file)
		subj, _ = pull_gifti_data(subj)
	n_rows, n_cols = subj.shape
	group_array = np.empty((n_rows*n_sub, n_cols), np.float64)
	return group_array


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
	# os.system(f'./merge_metric_resample.sh {script_name}.L.func.gii '
	# 	f'{script_name}.R.func.gii {script_name}_results.dtseries.nii')













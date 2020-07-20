import boto3
import nibabel as nb 
import numpy as np
import os

from glob import glob
from io import BytesIO
from itertools import groupby 
from nibabel import FileHolder, Cifti2Image, GiftiImage 
from nibabel.gifti.gifti import GiftiDataArray


def assign_files_to_list(bucket_obj, input_type):
	if input_type == 'cifti':
		files = list(bucket_obj.objects.filter(Prefix='data/proc_2_norm_filter/'))
		subj_files = [obj for obj in files if obj.key.endswith('.nii')]
	elif input_type == 'gifti':
		files = list(bucket_obj.objects.filter(Prefix='data/proc_3_surf_resamp/'))
		subj_files = [obj for obj in files if obj.key.endswith('.gii')]
		subj_files = assign_gifti_files_to_nested_list(subj_files)
	return subj_files


def assign_gifti_files_to_nested_list(gifti_files):
	id_extract = lambda x: x.split('/')[-1].split('_')[0]
	temp = sorted(gifti_files, key = id_extract) 
	nested_gifti_list = [list(subj_id) for i, subj_id in groupby(temp, id_extract)] 
	return nested_gifti_list


def get_subj_file_list(n_sub, input_type, aws=False, resource=None, bucket_name=None):
	if aws:
		my_bucket = resource.Bucket(bucket_name)
		subj_files = assign_files_to_list(my_bucket, input_type)
		file_key_dict = {file_obj.key:file_obj for file_obj in subj_files}
		subj_files = [file_key_dict[key] for key in sorted(file_key_dict.keys())]
	else:
		if input_type == 'cifti':
			subj_files = sorted(glob('data/proc_2_norm_filter/*dtseries.nii'))
		elif input_type == 'gifti':
			subj_files = glob('data/proc_3_surf_resamp/*.gii')
			subj_files = assign_gifti_files_to_nested_list(subj_files)
	if len(subj_files) < 1:
		raise Exception('No files found in file path')
	if n_sub is None:
		n_sub = len(subj_files)
	subj_files_sub = subj_files[:n_sub]
	return subj_files_sub


def load_cifti(cifti_fp, aws=False, client=None, bucket_name=None):
	if aws:
		cifti_obj = client.get_object(Bucket=bucket_name, Key=cifti_fp.key)
		cifti_bytes = FileHolder(fileobj=BytesIO(cifti_obj['Body'].read()))
		cifti = Cifti2Image.from_file_map(
			{'header': cifti_bytes, 
			'image': cifti_bytes}
			)
	else:
		cifti = nb.load(cifti_fp)
	return cifti


def load_gifti(gifti_fps, aws=False, client=None, bucket_name=None):
	gifti_LR = []
	file_base = gifti_fps[0].split('.')[0]
	if aws:
		for hem in ['L', 'R']:
			indx = gifti_fps.index(f'{file_base}.{hem}.func.gii')
			gifti_obj = client.get_object(Bucket=bucket_name, 
				Key=gifti_fps[indx].key)
			gifti_bytes = FileHolder(fileobj=BytesIO(gifti_obj['Body'].read()))
			gifti = GiftiImage.from_file_map(
				{'header': gifti_bytes,
				'image': gifti_bytes}
				)
			gifti_LR.append(gifti)
	else:
		for hem in ['L', 'R']:
			indx = gifti_fps.index(f'{file_base}.{hem}.func.gii')
			gifti_LR.append(nb.load(gifti_fps[indx]))
	return gifti_LR


def load_data_and_stack(n_sub, input_type, aws=False, bucket_name=None):
	if aws:
		# Ensure you have AWS CLI installed and it's properly 
		# configured - 'aws configure'
		client = boto3.client('s3')
		resource = boto3.resource('s3')
	else:
		client = None
		resource = None
	subj_files = get_subj_file_list(n_sub, input_type, aws, 
		resource, bucket_name)
	group_data = pre_allocate_array(subj_files[0], input_type, n_sub, 
		aws, client, bucket_name)
	row_indx = 0
	for subj_file in subj_files:
		print(subj_file)
		if input_type == 'cifti':
			subj_file = load_cifti(subj_file, aws, client, bucket_name)
			hdr, subj_data, n_time = pull_cifti_data(subj_file)
		elif input_type == 'gifti':
			subj_file = load_gifti(subj_file, aws, client, bucket_name)
			subj_data, n_time = pull_gifti_data(subj_file)
			hdr = subj_file
		group_data[row_indx:(row_indx+n_time), :] = subj_data
		row_indx += n_time
	return group_data, hdr


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


def pre_allocate_array(subj_file, input_type, n_sub, aws=False, 
	client=None, bucket_name=None):
	if input_type == 'cifti':
		subj = load_cifti(subj_file, aws, client, bucket_name)	
	elif input_type == 'gifti':
		subj = load_gifti(subj_file, aws, client, bucket_name)
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


def write_to_gifti(result, giftis, script_name):
	example_array_L = giftis[0].darrays[0]
	example_array_R = giftis[1].darrays[0]
	L_shape = len(giftis[0].agg_data()[0])
	L_result = result[:, :L_shape]
	R_result = result[:, L_shape:]
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













import boto3
import numpy as np

from io import BytesIO
from nibabel import FileHolder, Cifti2Image


# Ensure you have AWS CLI installed and it's properly configured - 'aws configure'


def assign_cifti_files_to_list(bucket_obj):
	files = list(bucket_obj.objects.filter(Prefix='data/'))
	cifti_files = [obj for obj in files if obj.key.endswith('.nii')]
	return cifti_files


def load_data_and_stack_s3(bucket_name, n_sub):
	client = boto3.client('s3')
	resource = boto3.resource('s3')
	my_bucket = resource.Bucket(bucket_name)
	cifti_files = assign_cifti_files_to_list(my_bucket)
	if n_sub is None:
		n_sub = len(cifti_files)
	cifti_files_sub = cifti_files[:n_sub]
	group_data = pre_allocate_array(client, cifti_files_sub, n_sub)
	row_indx=0
	for cifti_file in cifti_files_sub:
		print(cifti_file.key)
		cifti_obj = client.get_object(Bucket=bucket_name, Key=cifti_file.key)
		cifti_bytes = FileHolder(fileobj=BytesIO(cifti_obj['Body'].read()))
		cifti = Cifti2Image.from_file_map({'header': cifti_bytes, 'image': cifti_bytes})
		cifti.set_data_dtype('<f4')
		cifti_data = np.array(cifti.get_fdata())
		cifti.uncache()
		n_time = cifti.shape[0]
		group_data[row_indx:(row_indx+n_time), :] = cifti_data
		row_indx += n_time
	hdr = cifti.header
	return group_data, hdr


def pre_allocate_array(client_obj, cifti_files, n_sub):
	cifti_obj = client_obj.get_object(Bucket=bucket_name, Key=cifti_files[0].key)
	cifti_bytes = FileHolder(fileobj=BytesIO(cifti_obj['Body'].read()))
	cifti = Cifti2Image.from_file_map({'header': cifti_bytes, 'image': cifti_bytes})
	n_rows, n_cols = cifti.shape
	cifti_group_arr = np.zeros((n_rows*n_sub, n_cols), np.float64)

	return cifti_group_arr 








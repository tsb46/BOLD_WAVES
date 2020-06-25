import boto3

from io import BytesIO
from nibabel import FileHolder, Cifti2Image


# Ensure you have AWS CLI installed and its properly configured - 'aws configure'


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
	cifti_group = []
	cifti_files_sub = cifti_files[:n_sub]
	for cifti_file in cifti_files_sub:
		cifti_obj = client.get_object(Bucket=bucket_name, Key=cifti_file.key)
		print(cifti_file.key)
		cifti_bytes = FileHolder(fileobj=BytesIO(cifti_obj['Body'].read()))
		cifti = Cifti2Image.from_filemap({'header': cifti_bytes, 'image': cifti_bytes})
		cifti_data = np.array(cifti.get_fdata())
		cifti.uncache()
		cifti_group.append(cifti_data)
	group_data = np.concatenate(cifti_group, axis=0)
	hdr = cifti.header
	return group_data, hdr





#!/bin/bash

input_dir=$1
output_dir=$2
for filename in $input_dir/*.dtseries.nii; do
    echo "$filename" 
    file_base_dt_nii=$(basename -- "$filename")
    file_base_dt="${file_base_dt_nii%.*}"
    file_base="${file_base_dt%.*}"
    wb_command -cifti-smoothing $filename 2.12 2.12 COLUMN $output_dir/"$file_base"_smooth.dtseries.nii -left-surface templates/S900.L.midthickness_MSMAll.32k_fs_LR.surf.gii -right-surface templates/S900.R.midthickness_MSMAll.32k_fs_LR.surf.gii
done
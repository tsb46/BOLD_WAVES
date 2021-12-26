#!/bin/bash

input_dir=$1
parcellation=$2
output_dir=$3
for filename in $input_dir/*.dtseries.nii; do
    echo "$filename" 
    file_base_dt_nii=$(basename -- "$filename")
    file_base_dt="${file_base_dt_nii%.*}"
    file_base="${file_base_dt%.*}"
    wb_command -cifti-parcellate $filename $parcellation COLUMN $output_dir/${file_base}_parcel_ts.ptseries.nii
done
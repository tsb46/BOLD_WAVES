#!/bin/bash
input_dir=$1
output_dir=$2

for filename in $input_dir/*.dtseries.nii; do
    echo "$filename" 
    python preprocess/norm_filter.py -c $filename -l 0.01 -u 0.1 -o $output_dir
done
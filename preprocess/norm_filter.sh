#!/bin/bash

for filename in data/proc_1_smooth/*.dtseries.nii; do
    echo "$filename" 
    python preprocess/norm_filter.py -c $filename -l 0.01 -u 0.1 -o data/proc_2_norm_filter
done
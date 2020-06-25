#!/bin/bash

for filename in data/raw/*.dtseries.nii; do
    echo "$filename" 
    file_base_dt_nii=$(basename -- "$filename")
    file_base_dt="${file_base_dt_nii%.*}"
    file_base="${file_base_dt%.*}"
    wb_command -cifti-smoothing $filename 5.0 5.0 COLUMN data/proc_1_smooth/"$file_base"_smooth.dtseries.nii -left-surface templates/S900.L.midthickness_MSMAll.32k_fs_LR.surf.gii -right-surface templates/S900.R.midthickness_MSMAll.32k_fs_LR.surf.gii

done
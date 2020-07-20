#!/bin/bash

for filename in data/proc_2_norm_filter/*.dtseries.nii; do
    file_base_dt_nii=$(basename -- "$filename")
    file_base_dt="${file_base_dt_nii%.*}"
    file_base="${file_base_dt%.*}"
    echo "$file_base"
    ## 1. Separate out surface cortex from cifti
    wb_command -cifti-separate $filename COLUMN -metric CORTEX_LEFT \
    data/proc_3_surf_resamp/"$file_base"_L.func.gii \
    -metric CORTEX_RIGHT \
    data/proc_3_surf_resamp/"$file_base"_R.func.gii

    ## 2. Resample left and right surface metric files
    # 2.1 Resample right
    wb_command -metric-resample data/proc_3_surf_resamp/$"$file_base"_R.func.gii \
    templates/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii \
    templates/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii ADAP_BARY_AREA \
    data/proc_3_surf_resamp/"$file_base"_smooth_filt_resamp.R.func.gii \
    -area-metrics templates/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii \
    templates/fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii
    # 2.2. Resample left
    wb_command -metric-resample data/proc_3_surf_resamp/$"$file_base"_L.func.gii \
    templates/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
    templates/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii ADAP_BARY_AREA \
    data/proc_3_surf_resamp/"$file_base"_smooth_filt_resamp.L.func.gii \
    -area-metrics templates/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii \
    templates/fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii

    # 3. Clean up disk
    rm data/proc_3_surf_resamp/$"$file_base"_R.func.gii
    rm data/proc_3_surf_resamp/$"$file_base"_L.func.gii

done
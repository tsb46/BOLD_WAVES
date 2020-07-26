#!/bin/bash
# This script exists to reverse the resampling step in the preprocessing - 
# for visualization of results as cifti files in connectome workbench

res_left=$1
res_right=$2
output=$3
## 1. Resample left and right surface metric result files 
# back to original res - 32k per vertex
# 2.1 Resample right
wb_command -metric-resample $res_right \
templates/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
templates/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA \
"$res_right"_resamp.R.func.gii \
-area-metrics  templates/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii \
templates/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii
# 2.2. Resample left
wb_command -metric-resample $res_left \
templates/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
templates/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA \
"$res_left"_resamp.L.func.gii \
-area-metrics  templates/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii \
templates/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii

## 2. Merge metric files
wb_command -cifti-create-dense-timeseries $output \
-left-metric "$res_right"_resamp.R.func.gii \
-right-metric "$res_left"_resamp.L.func.gii

# 3. Clean up disk
rm $res_left
rm $res_right
rm $"res_left"_resamp.L.func.gii
rm $"res_right"_resamp.R.func.gii


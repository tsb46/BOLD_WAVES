#!/bin/bash
# Make directories
mkdir -p data/rest/proc_1_smooth
mkdir -p data/rest/proc_2_norm_filter
mkdir -p data/rest/proc_3_surf_resamp
mkdir -p data/rest/proc_4_gs_regress
mkdir -p data/rest/proc_5_parcel_ts

# Smooth 
# echo "smoothing..." 
# ./preprocess/surface_smooth.sh data/rest/raw data/rest/proc_1_smooth
# # Normalize (zscore) and filter
# echo "norming and filtering..."
# preprocess/norm_filter.sh data/rest/proc_1_smooth data/rest/proc_2_norm_filter
# # Resample to fsaverage4
# echo "resampling..."
# preprocess/fsaverage_resample.sh data/rest/proc_2_norm_filter data/rest/proc_3_surf_resamp
# Remove global signal
# echo "removing global signal..."
# preprocess/global_signal_regress.sh data/rest/proc_3_surf_resamp data/rest/proc_4_gs_regress
# Parcellate time series from cifti files
echo "parcellating cortex"
preprocess/parcel_ts.sh data/rest/proc_2_norm_filter templates/Glasser_Parcellation_210.32k_fs_LR.dlabel.nii data/rest/proc_5_parcel_ts





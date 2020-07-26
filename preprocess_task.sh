#!/bin/bash
# Make directories
mkdir -p data/task/raw
mkdir -p data/task/proc_1_smooth
mkdir -p data/task/proc_2_norm_filter
mkdir -p data/task/proc_3_surf_resamp
mkdir -p data/task/proc_4_gs_regress

# Smooth 
./preprocess/surface_smooth.sh data/task/raw data/task/proc_1_smooth
# Normalize (zscore) and filter
preprocess/norm_filter.sh data/task/proc_1_smooth data/task/proc_2_norm_filter
# Resample to fsaverage4
preprocess/fsaverage_resample.sh data/task/proc_2_norm_filter data/task/proc_3_surf_resamp


#!/bin/bash

# 1. Working Memory Task
# Make directories
# mkdir -p data/task/wm/raw
# mkdir -p data/task/wm/proc_1_smooth
# mkdir -p data/task/wm/proc_2_norm_filter
# mkdir -p data/task/wm/proc_3_surf_resamp
# mkdir -p data/task/wm/proc_4_gs_regress

# # Smooth 
# ./preprocess/surface_smooth.sh data/task/wm/raw data/task/wm/proc_1_smooth
# # Normalize (zscore) and filter
# preprocess/norm_filter.sh data/task/wm/proc_1_smooth data/task/wm/proc_2_norm_filter
# # Resample to fsaverage4
# preprocess/fsaverage_resample.sh data/task/wm/proc_2_norm_filter data/task/wm/proc_3_surf_resamp

# 2. rel task
# Make directories
mkdir -p data/task/rel/raw
mkdir -p data/task/rel/proc_1_smooth
mkdir -p data/task/rel/proc_2_norm_filter
mkdir -p data/task/rel/proc_3_surf_resamp
mkdir -p data/task/rel/proc_4_gs_regress

# Smooth 
./preprocess/surface_smooth.sh data/task/rel/raw data/task/rel/proc_1_smooth
# Normalize (zscore) and filter
preprocess/norm_filter.sh data/task/rel/proc_1_smooth data/task/rel/proc_2_norm_filter
# Resample to fsaverage4
preprocess/fsaverage_resample.sh data/task/rel/proc_2_norm_filter data/task/rel/proc_3_surf_resamp
#!/bin/bash

input_dir=$1
output_dir=$2

subj_nums=$(find "$input_dir"/*.gii -maxdepth 1 -exec basename "{}" \; | cut -d'_' -f1 | sort -u)

for subj_num in $subj_nums; do
    gifti_files=$(find "$input_dir"/$subj_num* | sort -u)
    echo $gifti_files
    gifti_L=$(echo $gifti_files | cut -d' ' -f1)
    gifti_R=$(echo $gifti_files | cut -d' ' -f2)
    python preprocess/global_signal_regress.py -l $gifti_L -r $gifti_R -o $output_dir
done
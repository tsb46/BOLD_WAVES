#!/bin/bash

subj_nums=$(find data/proc_3_surf_resamp/*.gii -maxdepth 1 -exec basename "{}" \; | cut -d'_' -f1 | sort -u)

for subj_num in $subj_nums; do
    gifti_files=$(find data/proc_3_surf_resamp/$subj_num* | sort -u)
    echo $gifti_files
    gifti_L=$(echo $gifti_files | cut -d' ' -f1)
    gifti_R=$(echo $gifti_files | cut -d' ' -f2)
    python preprocess/global_signal_regress.py -l $gifti_L -r $gifti_R -o data/proc_4_gs_regress
done
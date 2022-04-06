#!/bin/bash

# echo "Run Full Analysis"

# # Complex PCA and Temporal Reconstruction (Figure 2)
# echo "cpca"
# python run_main_pca.py -n 3 -p complex

# echo "temporal reconstruction of cpca components"
# python run_cpca_reconstruction.py -i pca_complex_results.pkl -n 3 

# # FC Survey Analysis (Figure 3)
# echo "pca"
# python run_main_pca.py -n 3

# echo "pca w/ varimax rotation"
# python run_main_pca.py -n 3 -r varimax

# echo "eigenmap (primary FC gradient)"
# python run_eigenmap.py -n 1 -p 90

# echo "spatial ica"
# python run_ica.py -n 3 -type spatial
# mv ica.dtseries.nii s_ica.dtseries.nii
# mv ica_results.pkl s_ica_results.pkl

# echo "temporal ica"
# python run_ica.py -n 3 -type temporal
# mv ica.dtseries.nii t_ica.dtseries.nii
# mv ica_results.pkl t_ica_results.pkl

# echo "hidden markov model"
# python run_hmm.py -n 3

# echo "caps - precuneus"
# python run_cap_analysis.py -lv 1794 -rv 1384 -n 2 -m 0
# mv caps.dtseries.nii caps_prec.dtseries.nii
# mv caps_results.pkl caps_prec_results.pkl

# echo "caps - somatosensory"
# python run_cap_analysis.py -lv 774 -rv 929 -n 2 -m 0
# mv caps.dtseries.nii caps_sm.dtseries.nii
# mv caps_results.pkl caps_sm_results.pkl

# echo "caps - supramarginal gyrus"
# python run_cap_analysis.py -lv 1371 -rv 1789 -n 2 -m 0
# mv caps.dtseries.nii caps_smg.dtseries.nii
# mv caps_results.pkl caps_smg_results.pkl

# echo "seed based fc - precuneus"
# python run_seed_fc.py -lv 1794 -rv 1384 
# mv fc_map.dtseries.nii fc_map_prec.dtseries.nii
# mv fc_map_results.pkl fc_map_prec_results.pkl

# echo "seed based fc - somatosensory"
# python run_seed_fc.py -lv 774 -rv 929
# mv fc_map.dtseries.nii fc_map_sm.dtseries.nii
# mv fc_map_results.pkl fc_map_sm_results.pkl

# echo "seed based fc - supramarginal gyrus"
# python run_seed_fc.py -lv 1371 -rv 1789 
# mv fc_map.dtseries.nii fc_map_smg.dtseries.nii
# mv fc_map_results.pkl fc_map_smg_results.pkl

# Movie 2 (quasiperiodic pattern and global signal)
echo "quasiperiodic pattern"
python run_qpp.py -s 50

echo "global signal analysis - peak averaging, beta map"
python run_global_signal_analysis.py

# Figure 4 (Lag Projection analysis)
echo "lag projection"
python run_lag_projection.py 

# Figure 5 (Comparison of primary FC gradient, pattern two and TP/TN pattern)
echo "pca - time-point (row) centered"
python run_main_pca.py -n 3 -c r

echo "eigenmaps across thresholds"
for p in 0 10 20 30 40 50 60 70 90
do
   echo "eigenmap $p pct. threshold"
   python run_eigenmap.py -n 1 -p $p
   mv eigenmap.dtseries.nii "eigenmap_${p}_thres.dtseries.nii"
   rm eigenmap_results.dtseries.nii
done

# Figure 6 
echo "compute FC matrix of all voxel time courses"
python run_fc_matrix.py 





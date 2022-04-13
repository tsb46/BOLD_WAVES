# A Parsimonious Description of Global Functional Brain Organization in Three Spatiotemporal Patterns
This repository contains the code and analyses for a forthcoming paper titled 'A Parsimonious Description of Global Functional Brain Organization in Three Spatiotemporal Patterns'.


## Article Abstract
Resting-state functional MRI has yielded many, seemingly disparate insights into large-scale organization of the human brain. Representations of the brain’s large-scale organization can be divided into two broad categories - zero-lag representations of functional connectivity structure and time-lag representations of traveling wave or propagation structure. Here we sought to unify multiple observed phenomena in these two categories by applying concepts from the study of standing (zero-lag) and traveling (time-lag) waves. Using simulated data, we first described the behavior of zero-lag and time-lag analyses applied to spatiotemporal patterns with varying degrees of standing and traveling wave patterns. We then identified three low-frequency spatiotemporal patterns in empirical resting-state fMRI signals, composed of a mixture of standing and traveling wave dynamics, that describe both the zero-lag and time-lag structure of these signals. We showed that a wide range of previously observed empirical phenomena, including functional connectivity gradients, the task-positive/task-negative pattern, the global signal, time-lag propagation patterns, the quasiperiodic pattern, and the functional connectome network structure are manifestations of these three spatiotemporal patterns. These patterns account for much of the global spatial structure that underlies functional connectivity analyses, and therefore impact how we interpret everything derived from resting-state fMRI data from functional networks to graphs.

## Purpose of Code
This code repository exists to replicate the results of our article. This code is not meant to be a regularly-maintained library of functional neuroimaging methods. However, outside the loading and writing functions, the analysis code is modular and should be adaptable to other contexts. The figure and simulation notebooks require a large number of result outputs to run. In order to run these notebooks from scratch, we stored the necessary result outputs in the ```demo_files```. Thus, a full clone of the repo is about 200MB in size. We hope to find a more suitable place to publically host these files in the near future. 


# 1. Installing
* Note, some of these steps are only necessary to replicate the results in the study. If you're interested in specific components of the preprocessing and analysis pipeline, some of these steps are not necessary.
All of the code in this repo was run with Python 3.8. Not 100% sure code will work with other versions of Python 3.
Assuming Git has already been installed on your desktop, copy and paste the following code into your terminal to download the repo:
```
git clone https://github.com/tsb46/BOLD_WAVES.git
```
## Basic Dependencies
In the base directory of the repo, pip install all necessary packages. If you just want to run the command-line scripts, without the notebooks (figures and simulations), you can pip install the necessary packages with the following command in the terminal:

```
pip install -r requirements.txt
```

## Notebook Dependencies
Because the notebooks depend on a much larger number of packages, we have another more extensive requirements.txt file, ```requirements_notebook.txt```. If you want to run the code in the ```figures.ipynb``` and ```simulation.ipynb``` notebooks, you can pip install the necessary packages with the following command in the terminal:

```
pip install -r requirements_notebook.txt
```
**Note**: ```requirements_notebook.txt``` contains all the packages in the ```requirements.txt```. It is not necessary to run both commands.

## FMRI Pre-processing Dependencies
For fMRI preprocessing functions, one will also need to install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) and Connectome Workbench (https://www.humanconnectome.org/software/connectome-workbench). To download resting-state fMRI scans from the HCP AWS bucket we use *R*. Thus, one will need need to have a locally-installed R software environment to use the download script.  

# 2. Figure and Simulation Notebooks
For those interested in the code used to generate the figures and post-hoc probing of analysis outputs, the ```figures.ipynb``` can be used without running any of the steps below. The same is true for the```simulation.ipynb``` notebook that contains the code used for to generate the simulations in **Figure 1** of the manuscript. 

# 3. Downloading & Pre-processing HCP Resting-State Data

## Download Using *neurohcp* R script
There are many ways to get obtain resting-state fMRI data from the Human Connectome Project (assuming proper authorization has been obtained from the HCP group). We programatically downloaded ICA-FIX resting-state fMRI data from the HCP AWS bucket using the ```neurohcp``` R package (https://github.com/muschellij2/neurohcp). We only downloaded one of the LR and RL scans from the first scanning session for each subject. The scripts and HCP subject IDs for downloading the sample data can found in:

```
├── data
│   ├── get_rest_cifti.R
│   ├── subject_100_ids.csv
```
Note, the first 50 subject IDs were used as the primary sample. The last 50 subject IDs were used as a replication sample. Pulling HCP data using *neurohcp* requires that the user has created AWS credentials at https://db.humanconnectome.org/ - you should be supplied with an API access key and secret key. The access key and secrete key should be placed as parameters in the *make_aws_call* function on line 5 of the ```get_rest_cifti.R``` script:

```
make_aws_call(path_to_file = "/", 
              bucket = "hcp-openaccess",region = "us-east-1", access_key = "", 
              secret_key = "",
              lifetime_minutes = 5, query = NULL, verb = "GET", sign = TRUE)
```
## Download Using Other Source
One could just as well download the HCP fMRI data using other means (using the subject IDs in ```subject_ids_main.csv```). If you download HCP fMRI data by other means, simply create a sub-directory called *rest* in the *data* folder, followed by a sub-directory in *rest* called *raw*. In other words:
```
├── data
│   ├── get_rest_cifti.R
│   ├── subject_ids_main.csv
│   ├── subject_ids_replication.csv
│   ├── rest
│   │   ├── raw
```

## Pre-processing
A single bash script in the main directory ```preprocess_rest.sh``` runs the download and pre-processing pipeline used in the article. The pre-processing includes (in order): surface smoothing (5mm FWHM), z-score normalization and band-pass filtering (0.01-0.1Hz), resampling of cortical surface to fsaverage4 space, and global signal regression (only used for certain analyses). The bash script accepts one parameter - a binary flag to indicate whether the R script should be used to download HCP data - 1 (yes) or 0 (no). So, if you would like to download the HCP fMRI data (using the provided R script):

```
./preprocess_rest.sh 1
```

If you have downloaded the HCP fMRI through other sources (and completed the steps above):

```
./preprocess_rest.sh 0
```

**Note**: in some cases, you may need to the bash script executable by using the following code (in your terminal):

```
chmod u+x preprocess_rest.sh
```

# 3. Walk-Through of Analysis Code

## Command-Line Scripts
Command-line scripts performed analyses that operated directly on the resting-state fMRI scan time courses. All of these analyses are performed at the group level by temporal concatenation of subject scan time courses. All command-line scripts are meant to be run from the base directory of the repo. We provide a convenience bash shell script (```run_analysis.sh```) to run all the analyses in one command. Simply enter the following command into the terminal (while in the base directory of the repo):

```
./run_analysis.sh
```

For those interested in a particular analysis, the following scripts are provided in the base directory:

* ```run_main_pca.py``` - principal component analysis (PCA) and complex principal component analysis (cPCA). In addition, varimax rotation is implemented through an optional parameter. PCA is implemented using a fast-randomized SVD algorithm provided by the package ```fbpca```.
* ```run_cpca_reconstruction.py``` - this script takes a cPCA results pickle (.pkl) object from the output of ```run_main_pca.py```, and performs a temporal reconstruction of the cPCA components by binned averaging of the phase time course.
* ```run_eigenmap.py``` - implements the Laplacian Eigenmap (LE) algorithm on a vertex-by-vertex affinity matrix (cosine similarity), as in Vos de Waal et al. (2020). This script also implements a Kernel PCA on the vertex-by-vertex affinity matrix. Both the LE and Kernel PCA analysis algorithms are implemented using ```scikit-learn```.
* ```run_ica.py``` - implements temporal and spatial independent component analysis (ICA). Both analyses are implemented using the FastICA algorithm in ```scikit-learn```.
* ```run_cap_analysis.py``` - implments co-activation pattern (CAP) analysis for selected seed locations. K-Means clustering is performed on suprathreshold time points using the KMeans implemented in ```scikit-learn```. Seed locations must be provided by the vertex number in the fsaverage4 space for the right hemisphere and/or left hemisphere. These locations are specified using the ```-rv``` and ```-lv``` parameters, respectively. The main analyses performed CAP analysis on three seeds:
  * Precuneus: ```-lv 1794 -rv 1384```
  * Somatosensory: ```-lv 774 -rv 929```
  * Supramarginal gyrus: ```-lv 1371 -rv 1789```
* ```run_seed_fc.py``` - implements seed-based correlation analysis for selected seed locations. The same seeds from the CAP analysis are used, and the parameters to specify seed locations are identical.
* ```run_lag_projection.py``` - implements the lag projection algorithm described by Mitra et al. (2014). This is an inferior Python implementation of the excellent MATLAB implementation provided in https://github.com/ryraut/lag-code . Special thanks to the authors for making this code publically available.
* ```run_qpp.py``` - implements the template-averaging quasiperiodic (QPP) algorithm described in Majeed et al. (2009). This script was modified from the excellent implementation in the CPAC toolbox (https://github.com/BIDS-Apps/CPAC). Depending on the parameters (particularly, the number of repeats of the algorithm), this could take a while to run. To speed up the process, reduce the number of repeats of the algorithm (set at 10).
* ```run_hmm.py``` - implements a Gaussian Mixture Hidden Markov Model (GM-HMM) using the ```hmmlearn``` package. To ease the computational burden, PCA is performed as a dimension-reduction step before estimating the GM-HMM. 
* ```run_global_signal_analysis.py``` - this script performs some simple analytics on the mean global time course ('global signal'). First, it does a peak-averaing procedure to provide a spatiotemporal view of the pattern of BOLD activity around peaks of the global signal. Second, it performs a regression of each vertex time course on the global signal time course to visualize the relationship between the global signal time course and the time course at each vertex.
* ```run_peak_average.py``` - this is a generic script that performs peak averaging of BOLD activity around peaks of an arbitrary input time course, specified as a one-dimensional array in .txt format.


# References

Majeed, W., Magnuson, M., Hasenkamp, W., Schwarb, H., Schumacher, E. H., Barsalou, L., & Keilholz, S. D. (2011). Spatiotemporal dynamics of low frequency BOLD fluctuations in rats and humans. NeuroImage, 54(2), 1140–1150. https://doi.org/10.1016/j.neuroimage.2010.08.030

Mitra, A., Snyder, A. Z., Hacker, C. D., & Raichle, M. E. (2014). Lag structure in resting-state fMRI. Journal of Neurophysiology, 111(11), 2374–2391. https://doi.org/10.1152/jn.00804.2013

Vos de Wael, R., Benkarim, O., Paquola, C., Lariviere, S., Royer, J., Tavakol, S., Xu, T., Hong, S.-J., Langs, G., Valk, S., Misic, B., Milham, M., Margulies, D., Smallwood, J., & Bernhardt, B. C. (2020). BrainSpace: A toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. Communications Biology, 3(1), 1–10. https://doi.org/10.1038/s42003-020-0794-7





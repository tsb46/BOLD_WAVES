# A Parsimonious Description of Global Functional Brain Organization in Three Spatiotemporal Patterns
This repository contains the code and analyses for a forthcoming paper titled 'A Parsimonious Description of Global Functional Brain Organization in Three Spatiotemporal Patterns'.


## Article Abstract
Resting-state functional MRI has yielded many, seemingly disparate insights into large-scale organization of the human brain. Representations of the brain’s large-scale organization can be divided into two broad categories - zero-lag representations of functional connectivity structure and time-lag representations of traveling wave or propagation structure. Here we sought to unify multiple observed phenomena in these two categories by applying concepts from the study of standing (zero-lag) and traveling (time-lag) waves. Using simulated data, we first described the behavior of zero-lag and time-lag analyses applied to spatiotemporal patterns with varying degrees of standing and traveling wave patterns. We then identified three low-frequency spatiotemporal patterns in empirical resting-state fMRI signals, composed of a mixture of standing and traveling wave dynamics, that describe both the zero-lag and time-lag structure of these signals. We showed that a wide range of previously observed empirical phenomena, including functional connectivity gradients, the task-positive/task-negative pattern, the global signal, time-lag propagation patterns, the quasiperiodic pattern, and the functional connectome network structure are manifestations of these three spatiotemporal patterns. These patterns account for much of the global spatial structure that underlies functional connectivity analyses, and therefore impact how we interpret everything derived from resting-state fMRI data from functional networks to graphs.

## Purpose of Code
This code repository exists to replicate the results of our article. This code is not meant to be a regularly-maintained library of functional neuroimaging methods. However, outside the loading and writing functions, the analysis code is modular and should be adaptable to other contexts. Because our article surveyed a large number of analytic methods, and many of these methods were implemented using other open-source Python packages, there is a long list of dependencies needed to replicate all the results in the article. If one is interested in a subset of the methods used in the article, one should be able to work with a smaller number of dependencies. Further, the figure and simulation notebooks require a large number of result outputs to run (in the 'demo_files' directory). Thus, a full clone of the repo is about 200MB in size. We hope to find a more suitable place to publically host these files in the near future. 

# 1. Installing
* Note, some of these steps are only necessary to replicate the results in the study. If you're interested in specific components of the preprocessing and analysis pipeline, some of these steps are not necessary.
All of the code in this repo was run with Python 3.8. Not 100% sure code will work with other versions of Python 3.
Assuming Git has already been installed on your desktop, copy and paste the following code into your terminal to download the repo:
```
git clone https://github.com/tsb46/BOLD_WAVES.git
```
Then, in the base directory of the repo, pip install all necessary packages (in requirements.txt) with the following command in the terminal:

```
pip install -r requirements.txt
```
For fMRI preprocessing functions, one will also need to install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) and Connectome Workbench (https://www.humanconnectome.org/software/connectome-workbench). 

# 2. Downloading HCP Resting-State Data
There are many ways to get obtain resting-state fMRI data from the Human Connectome Project (assuming proper authorization has been obtained from the HCP group). We programatically downloaded ICA-FIX resting-state fMRI data using the *neurohcp* R package. The scripts and HCP subject IDs for downloading the sample data can found in:

```
├── data
│   ├── get_rest_cifti.R
│   ├── subject_100_ids.csv
```
Note, the first 50 subject IDs were used as the primary sample. The last 50 subject IDs were used as a replication sample. Running the get_rest


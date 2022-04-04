if (!require("neurohcp")) {
  install.packages("neurohcp", repos="http://cran.rstudio.com/") 
  library("neurohcp")
}
make_aws_call(path_to_file = "/", 
              bucket = "hcp-openaccess",region = "us-east-1", access_key = "", 
              secret_key = "",
              lifetime_minutes = 5, query = NULL, verb = "GET", sign = TRUE)
subject_data<- read.csv('subject_100_ids.csv',header = TRUE)
subjects <- subject_data$Subject

output_dir = 'rest/raw'

#ICA-FIX data
for (n in 1:length(subjects)) {
  i = subjects[n]
  print(paste(n,': ',i))
  if (n %% 2 == 0) {
    download_hcp_file(paste("HCP_1200/",i,"/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii",sep=""), destfile = paste(output_dir,'/',i,"_RL1_rest.dtseries.nii", sep = ""), error=FALSE)
  }
  else {
    download_hcp_file(paste("HCP_1200/",i,"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii",sep=""), destfile = paste(output_dir,'/',i,"_LR1_rest.dtseries.nii", sep = ""), error=FALSE)
  }
}

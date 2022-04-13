if (!require("neurohcp")) {
  install.packages("neurohcp", repos="http://cran.rstudio.com/") 
  library("neurohcp")
}
make_aws_call(path_to_file = "/", 
              bucket = "hcp-openaccess",region = "us-east-1", access_key = "", 
              secret_key = "",
              lifetime_minutes = 5, query = NULL, verb = "GET", sign = TRUE)
subject_data<- read.csv('subject_ids_main.csv',header = TRUE)
subjects <- subject_data$Subject
scans <- subject_data$Scan

output_dir = 'rest/raw'

#ICA-FIX data
for (n in 1:length(subjects)) {
  i = subjects[n]
  s = scans[n]
  print(paste(n,': ',i))
  download_hcp_file(
    paste("HCP_1200/",i,"/MNINonLinear/Results/rfMRI_REST1_", s, "/rfMRI_REST1_", s, "_Atlas_MSMAll_hp2000_clean.dtseries.nii", sep=""), 
    destfile = paste(output_dir,'/',i,"_LR1_rest.dtseries.nii", sep = ""), error=FALSE)
}

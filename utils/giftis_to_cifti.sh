#!/bin/bash

gifti_L=$1
gifti_R=$2
output=$3

wb_command -cifti-create-dense-timeseries $output \
-left-metric $gifti_L -right-metric $gifti_R

rm $gifti_L
rm $gifti_R

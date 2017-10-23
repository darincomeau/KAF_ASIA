## Synopsis

This is code to run kernel analog forecasting to predict Arctic sea ice anomalies. Code for NLSA and access to data files needed.

## Files
dowork.m - Main driver file to run all analysis

all_regions_work.m - Called by dowork.m to run analysis on all regions

calc_ica_pred.m - Main file to run all steps of analysis to create the forecasts, including extracting data from .nc files, calculating time series for training and test periods, evaluating the kernel using NLSA, and creating forecasts.

ccsm4DataRaw.m - Script to extract data from netcdf file.

nlsa_ose_ica.m - Driver script for NLSA kernel evaluation.

LP_*.m - Scripts to calculate forecasts based on Laplacian pyramids.

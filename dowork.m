% driver script to run all analysis

clear all
close all

predOn = 1;
fullDataOn = 1;

trainLim = [100 499];
testLim  = [500 899];

varsUsed = 'SIC'
embedWin = 12
all_regions_work

varsUsed = 'SIC_SST'
embedWin = 12
all_regions_work

varsUsed = 'SIC_SST_SIT'
embedWin = 12
all_regions_work

varsUsed = 'SIC_SST_SIT_SLP'
embedWin = 12
all_regions_work

varsUsed = 'SIC_SIT'
embedWin = 12
all_regions_work

varsUsed = 'SIC_SLP'
embedWin = 12
all_regions_work

% one test with short training period
% dcnote - this will break with current manual loading of distances to form P
% trainLim = [100 139];
% testLim  = [500 899];
% region = 'Arctic'
% calc_ica_pred
% calc_iva_pred

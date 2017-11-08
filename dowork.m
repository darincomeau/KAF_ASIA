% driver script to run all analysis

clear all
close all

% predOn = 1;
% fullDataOn = 0;

% trainLim = [100 499];
% testLim  = [500 899];

% varsUsed = 'SIC'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SST'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SST_SIT'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SST_SIT_SLP'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SIT'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SLP'
% embedWin = 12
% all_regions_work

% varsUsed = 'SIC_SST_SLP'
% embedWin = 12
% all_regions_work

% % aggregate data
% comp_data('SIC',12,'ica',fullDataOn)
% comp_data('SIC',12,'iva',fullDataOn)
% comp_data('SIC_SST',12,'ica',fullDataOn)
% comp_data('SIC_SST',12,'iva',fullDataOn)
% comp_data('SIC_SST_SIT',12,'ica',fullDataOn)
% comp_data('SIC_SST_SIT',12,'iva',fullDataOn)
% comp_data('SIC_SST_SIT_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SST_SIT_SLP',12,'iva',fullDataOn)
% comp_data('SIC_SIT',12,'ica',fullDataOn)
% comp_data('SIC_SIT',12,'iva',fullDataOn)
% comp_data('SIC_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SLP',12,'iva',fullDataOn)
% comp_data('SIC_SST_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SST_SLP',12,'iva',fullDataOn)

% fullDataOn = 1;
% comp_data('SIC',12,'ica',fullDataOn)
% comp_data('SIC',12,'iva',fullDataOn)
% comp_data('SIC_SST',12,'ica',fullDataOn)
% comp_data('SIC_SST',12,'iva',fullDataOn)
% comp_data('SIC_SST_SIT',12,'ica',fullDataOn)
% comp_data('SIC_SST_SIT',12,'iva',fullDataOn)
% comp_data('SIC_SST_SIT_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SST_SIT_SLP',12,'iva',fullDataOn)
% comp_data('SIC_SIT',12,'ica',fullDataOn)
% comp_data('SIC_SIT',12,'iva',fullDataOn)
% comp_data('SIC_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SLP',12,'iva',fullDataOn)
% comp_data('SIC_SST_SLP',12,'ica',fullDataOn)
% comp_data('SIC_SST_SLP',12,'iva',fullDataOn)

% one test with longer embedding window for SIT
% predOn = 1;
% fullDataOn = 0;
% trainLim = [100 499];
% testLim  = [500 899];
% varsUsed = 'SIC_SST_SITq48'
% embedWin = 12
% region = 'Arctic'
% calc_ica_pred
% calc_iva_pred
 
% one test with short training period
% dcnote - this will break with current manual loading of distances to form P
% predOn = 1;
% fullDataOn = 0;
% trainLim = [100 499];
% testLim  = [500 899];
% varsUsed = 'SIC_SST_SIT'
% embedWin = 12
% trainLim = [100 139];
% testLim  = [500 899];
% region = 'Arctic'
% calc_ica_pred
% calc_iva_pred


predOn = 1;
fullDataOn = 0;

trainLim = [100 499];
testLim  = [500 899];
varsUsed = 'SIC_SST_SITq48'
embedWin = 12
all_regions_work
comp_data('SIC_SST_SITq48',12,'ica',fullDataOn)
comp_data('SIC_SST_SITq48',12,'iva',fullDataOn)







 
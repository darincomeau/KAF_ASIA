function comp_data(varsUsed,embedWin,iceVar)

	compDataDir = fullfile(strcat('output/',iceVar,'/',varsUsed,'_q',num2str(embedWin),'/'));
	if exist(compDataDir) == 0
    	mkdir(compDataDir)
	end

	% load data
	region =  'Arctic';
	lonLim = [0 360];
	latLim = [45 90];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_1_pcIM     = pred_pcIM;
	pred_panel_1_pcTM     = pred_pcTM;
	pred_panel_1_pcIMdiff = pred_pcIM;
	pred_panel_1_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_1_pcIMdiff = pred_panel_1_pcIMdiff - pred_pcIMP;
	pred_panel_1_pcTMdiff = pred_pcTM;
	pred_panel_1_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_1_pcTMdiff = pred_panel_1_pcTMdiff - pred_pcTMP;
	pred_panel_1_rms      = pred_rms;
	pred_panel_1_rmsP     = pred_rmsP;
	pred_panel_1_pc       = pred_pc;
	pred_panel_1_pcP      = pred_pcP;
	pred_panel_1_truth    = data_test;
	pred_panel_1_ose      = f_ext;
	    
	region = 'ChukchiBeaufort';
	lonLim = [175 235];
	latLim = [65 75];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_2_pcIM     = pred_pcIM;
	pred_panel_2_pcTM     = pred_pcTM;
	pred_panel_2_pcIMdiff = pred_pcIM;
	pred_panel_2_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_2_pcIMdiff = pred_panel_2_pcIMdiff - pred_pcIMP;
	pred_panel_2_pcTMdiff = pred_pcTM;
	pred_panel_2_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_2_pcTMdiff = pred_panel_2_pcTMdiff - pred_pcTMP;
	pred_panel_2_rms      = pred_rms;
	pred_panel_2_rmsP     = pred_rmsP;
	pred_panel_2_pc       = pred_pc;
	pred_panel_2_pcP      = pred_pcP;
	pred_panel_2_truth    = data_test;
	pred_panel_2_ose      = f_ext;
	        
	region = 'Chukchi';
	lonLim = [175 205];
	latLim = [65 75];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_3_pcIM     = pred_pcIM;
	pred_panel_3_pcTM     = pred_pcTM;
	pred_panel_3_pcIMdiff = pred_pcIM;
	pred_panel_3_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_3_pcIMdiff = pred_panel_3_pcIMdiff - pred_pcIMP;
	pred_panel_3_pcTMdiff = pred_pcTM;
	pred_panel_3_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_3_pcTMdiff = pred_panel_3_pcTMdiff - pred_pcTMP;
	pred_panel_3_rms      = pred_rms;
	pred_panel_3_rmsP     = pred_rmsP;
	pred_panel_3_pc       = pred_pc;
	pred_panel_3_pcP      = pred_pcP;
	pred_panel_3_truth    = data_test;
	pred_panel_3_ose      = f_ext;
	        
	region = 'Beaufort';
	lonLim = [205 235];
	latLim = [65 75];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_4_pcIM     = pred_pcIM;
	pred_panel_4_pcTM     = pred_pcTM;
	pred_panel_4_pcIMdiff = pred_pcIM;
	pred_panel_4_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_4_pcIMdiff = pred_panel_4_pcIMdiff - pred_pcIMP;
	pred_panel_4_pcTMdiff = pred_pcTM;
	pred_panel_4_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_4_pcTMdiff = pred_panel_4_pcTMdiff - pred_pcTMP;
	pred_panel_4_rms      = pred_rms;
	pred_panel_4_rmsP     = pred_rmsP;
	pred_panel_4_pc       = pred_pc;
	pred_panel_4_pcP      = pred_pcP;
	pred_panel_4_truth    = data_test;
	pred_panel_4_ose      = f_ext;

	region = 'EastSibLaptev';
	lonLim = [105 175];
	latLim = [65 75];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_5_pcIM     = pred_pcIM;
	pred_panel_5_pcTM     = pred_pcTM;
	pred_panel_5_pcIMdiff = pred_pcIM;
	pred_panel_5_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_5_pcIMdiff = pred_panel_5_pcIMdiff - pred_pcIMP;
	pred_panel_5_pcTMdiff = pred_pcTM;
	pred_panel_5_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_5_pcTMdiff = pred_panel_5_pcTMdiff - pred_pcTMP;
	pred_panel_5_rms      = pred_rms;
	pred_panel_5_rmsP     = pred_rmsP;
	pred_panel_5_pc       = pred_pc;
	pred_panel_5_pcP      = pred_pcP;
	pred_panel_5_truth    = data_test;
	pred_panel_5_ose      = f_ext;
	        
	region = 'EastSib';
	lonLim = [140 175];
	latLim = [65 75];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_6_pcIM     = pred_pcIM;
	pred_panel_6_pcTM     = pred_pcTM;
	pred_panel_6_pcIMdiff = pred_pcIM;
	pred_panel_6_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_6_pcIMdiff = pred_panel_6_pcIMdiff - pred_pcIMP;
	pred_panel_6_pcTMdiff = pred_pcTM;
	pred_panel_6_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_6_pcTMdiff = pred_panel_6_pcTMdiff - pred_pcTMP;
	pred_panel_6_rms      = pred_rms;
	pred_panel_6_rmsP     = pred_rmsP;
	pred_panel_6_pc       = pred_pc;
	pred_panel_6_pcP      = pred_pcP;
	pred_panel_6_truth    = data_test;
	pred_panel_6_ose      = f_ext;
	        
	region = 'Laptev';
	lonLim = [105 140];
	latLim = [70 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_7_pcIM     = pred_pcIM;
	pred_panel_7_pcTM     = pred_pcTM;
	pred_panel_7_pcIMdiff = pred_pcIM;
	pred_panel_7_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_7_pcIMdiff = pred_panel_7_pcIMdiff - pred_pcIMP;
	pred_panel_7_pcTMdiff = pred_pcTM;
	pred_panel_7_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_7_pcTMdiff = pred_panel_7_pcTMdiff - pred_pcTMP;
	pred_panel_7_rms      = pred_rms;
	pred_panel_7_rmsP     = pred_rmsP;
	pred_panel_7_pc       = pred_pc;
	pred_panel_7_pcP      = pred_pcP;
	pred_panel_7_truth    = data_test;
	pred_panel_7_ose      = f_ext;

	region = 'BarentsKara';
	lonLim = [30 95];
	latLim = [65 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_8_pcIM     = pred_pcIM;
	pred_panel_8_pcTM     = pred_pcTM;
	pred_panel_8_pcIMdiff = pred_pcIM;
	pred_panel_8_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_8_pcIMdiff = pred_panel_8_pcIMdiff - pred_pcIMP;
	pred_panel_8_pcTMdiff = pred_pcTM;
	pred_panel_8_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_8_pcTMdiff = pred_panel_8_pcTMdiff - pred_pcTMP;
	pred_panel_8_rms      = pred_rms;
	pred_panel_8_rmsP     = pred_rmsP;
	pred_panel_8_pc       = pred_pc;
	pred_panel_8_pcP      = pred_pcP;
	pred_panel_8_truth    = data_test;
	pred_panel_8_ose      = f_ext;
	        
	region = 'Barents';
	lonLim = [30 60];
	latLim = [65 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_9_pcIM     = pred_pcIM;
	pred_panel_9_pcTM     = pred_pcTM;
	pred_panel_9_pcIMdiff = pred_pcIM;
	pred_panel_9_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_9_pcIMdiff = pred_panel_9_pcIMdiff - pred_pcIMP;
	pred_panel_9_pcTMdiff = pred_pcTM;
	pred_panel_9_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_9_pcTMdiff = pred_panel_9_pcTMdiff - pred_pcTMP;
	pred_panel_9_rms      = pred_rms;
	pred_panel_9_rmsP     = pred_rmsP;
	pred_panel_9_pc       = pred_pc;
	pred_panel_9_pcP      = pred_pcP;
	pred_panel_9_truth    = data_test;
	pred_panel_9_ose      = f_ext;
	        
	region = 'Kara';
	lonLim = [60 95];
	latLim = [65 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_10_pcIM     = pred_pcIM;
	pred_panel_10_pcTM     = pred_pcTM;
	pred_panel_10_pcIMdiff = pred_pcIM;
	pred_panel_10_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_10_pcIMdiff = pred_panel_10_pcIMdiff - pred_pcIMP;
	pred_panel_10_pcTMdiff = pred_pcTM;
	pred_panel_10_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_10_pcTMdiff = pred_panel_10_pcTMdiff - pred_pcTMP;
	pred_panel_10_rms      = pred_rms;
	pred_panel_10_rmsP     = pred_rmsP;
	pred_panel_10_pc       = pred_pc;
	pred_panel_10_pcP      = pred_pcP;
	pred_panel_10_truth    = data_test;
	pred_panel_10_ose      = f_ext;

	region = 'Greenland';
	lonLim = [325 360];
	latLim = [65 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_11_pcIM     = pred_pcIM;
	pred_panel_11_pcTM     = pred_pcTM;
	pred_panel_11_pcIMdiff = pred_pcIM;
	pred_panel_11_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_11_pcIMdiff = pred_panel_11_pcIMdiff - pred_pcIMP;
	pred_panel_11_pcTMdiff = pred_pcTM;
	pred_panel_11_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_11_pcTMdiff = pred_panel_11_pcTMdiff - pred_pcTMP;
	pred_panel_11_rms      = pred_rms;
	pred_panel_11_rmsP     = pred_rmsP;
	pred_panel_11_pc       = pred_pc;
	pred_panel_11_pcP      = pred_pcP;
	pred_panel_11_truth    = data_test;
	pred_panel_11_ose      = f_ext;

	region = 'Baffin';
	lonLim = [280 310];
	latLim = [70 80];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_12_pcIM     = pred_pcIM;
	pred_panel_12_pcTM     = pred_pcTM;
	pred_panel_12_pcIMdiff = pred_pcIM;
	pred_panel_12_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_12_pcIMdiff = pred_panel_12_pcIMdiff - pred_pcIMP;
	pred_panel_12_pcTMdiff = pred_pcTM;
	pred_panel_12_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_12_pcTMdiff = pred_panel_12_pcTMdiff - pred_pcTMP;
	pred_panel_12_rms      = pred_rms;
	pred_panel_12_rmsP     = pred_rmsP;
	pred_panel_12_pc       = pred_pc;
	pred_panel_12_pcP      = pred_pcP;
	pred_panel_12_truth    = data_test;
	pred_panel_12_ose      = f_ext;

	region = 'Labrador';
	lonLim = [290 310];
	latLim = [50 70];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_13_pcIM     = pred_pcIM;
	pred_panel_13_pcTM     = pred_pcTM;
	pred_panel_13_pcIMdiff = pred_pcIM;
	pred_panel_13_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_13_pcIMdiff = pred_panel_13_pcIMdiff - pred_pcIMP;
	pred_panel_13_pcTMdiff = pred_pcTM;
	pred_panel_13_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_13_pcTMdiff = pred_panel_13_pcTMdiff - pred_pcTMP;
	pred_panel_13_rms      = pred_rms;
	pred_panel_13_rmsP     = pred_rmsP;
	pred_panel_13_pc       = pred_pc;
	pred_panel_13_pcP      = pred_pcP;
	pred_panel_13_truth    = data_test;
	pred_panel_13_ose      = f_ext;

	region = 'Hudson';
	lonLim = [265 285];
	latLim = [55 65];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_14_pcIM     = pred_pcIM;
	pred_panel_14_pcTM     = pred_pcTM;
	pred_panel_14_pcIMdiff = pred_pcIM;
	pred_panel_14_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_14_pcIMdiff = pred_panel_14_pcIMdiff - pred_pcIMP;
	pred_panel_14_pcTMdiff = pred_pcTM;
	pred_panel_14_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_14_pcTMdiff = pred_panel_14_pcTMdiff - pred_pcTMP;
	pred_panel_14_rms      = pred_rms;
	pred_panel_14_rmsP     = pred_rmsP;
	pred_panel_14_pc       = pred_pc;
	pred_panel_14_pcP      = pred_pcP;
	pred_panel_14_truth    = data_test;
	pred_panel_14_ose      = f_ext;
	    
	region = 'Bering';
	lonLim = [165 200];
	latLim = [55 65];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_15_pcIM     = pred_pcIM;
	pred_panel_15_pcTM     = pred_pcTM;
	pred_panel_15_pcIMdiff = pred_pcIM;
	pred_panel_15_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_15_pcIMdiff = pred_panel_15_pcIMdiff - pred_pcIMP;
	pred_panel_15_pcTMdiff = pred_pcTM;
	pred_panel_15_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_15_pcTMdiff = pred_panel_15_pcTMdiff - pred_pcTMP;
	pred_panel_15_rms      = pred_rms;
	pred_panel_15_rmsP     = pred_rmsP;
	pred_panel_15_pc       = pred_pc;
	pred_panel_15_pcP      = pred_pcP;
	pred_panel_15_truth    = data_test;
	pred_panel_15_ose      = f_ext;
	        
	region = 'Okhotsk';
	lonLim = [135 165];
	latLim = [45 65];
	saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin));
	saveDir = fullfile('output/',saveTag,'/');
	S = fullfile(strcat(saveDir,'pred_',iceVar,'.mat'));
	load(S)
	pred_panel_16_pcIM     = pred_pcIM;
	pred_panel_16_pcTM     = pred_pcTM;
	pred_panel_16_pcIMdiff = pred_pcIM;
	pred_panel_16_pcIMdiff((pred_pcIM<0.5) & (pred_pcIMP<0.5)) = nan;
	pred_panel_16_pcIMdiff = pred_panel_16_pcIMdiff - pred_pcIMP;
	pred_panel_16_pcTMdiff = pred_pcTM;
	pred_panel_16_pcTMdiff((pred_pcTM<0.5) & (pred_pcTMP<0.5)) = nan;
	pred_panel_16_pcTMdiff = pred_panel_16_pcTMdiff - pred_pcTMP;
	pred_panel_16_rms      = pred_rms;
	pred_panel_16_rmsP     = pred_rmsP;
	pred_panel_16_pc       = pred_pc;
	pred_panel_16_pcP      = pred_pcP;
	pred_panel_16_truth    = data_test;
	pred_panel_16_ose      = f_ext;


	if embedWin == 6
	    mM = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5];
	elseif embedWin == 12 | embedWin == 24
	    mM = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
	end

	pred_panel_shift_1_pcIM     = pred_panel_1_pcIM(mM,:);
	pred_panel_shift_1_pcTM     = pred_panel_1_pcTM(mM,:);
	pred_panel_shift_1_pcIMdiff = pred_panel_1_pcIMdiff(mM,:);
	pred_panel_shift_1_pcTMdiff = pred_panel_1_pcTMdiff(mM,:);
	pred_panel_shift_2_pcIM     = pred_panel_2_pcIM(mM,:);
	pred_panel_shift_2_pcTM     = pred_panel_2_pcTM(mM,:);
	pred_panel_shift_2_pcIMdiff = pred_panel_2_pcIMdiff(mM,:);
	pred_panel_shift_2_pcTMdiff = pred_panel_2_pcTMdiff(mM,:);
	pred_panel_shift_3_pcIM     = pred_panel_3_pcIM(mM,:);
	pred_panel_shift_3_pcTM     = pred_panel_3_pcTM(mM,:);
	pred_panel_shift_3_pcIMdiff = pred_panel_3_pcIMdiff(mM,:);
	pred_panel_shift_3_pcTMdiff = pred_panel_3_pcTMdiff(mM,:);
	pred_panel_shift_4_pcIM     = pred_panel_4_pcIM(mM,:);
	pred_panel_shift_4_pcTM     = pred_panel_4_pcTM(mM,:);
	pred_panel_shift_4_pcIMdiff = pred_panel_4_pcIMdiff(mM,:);
	pred_panel_shift_4_pcTMdiff = pred_panel_4_pcTMdiff(mM,:);
	pred_panel_shift_5_pcIM     = pred_panel_5_pcIM(mM,:);
	pred_panel_shift_5_pcTM     = pred_panel_5_pcTM(mM,:);
	pred_panel_shift_5_pcIMdiff = pred_panel_5_pcIMdiff(mM,:);
	pred_panel_shift_5_pcTMdiff = pred_panel_5_pcTMdiff(mM,:);
	pred_panel_shift_6_pcIM     = pred_panel_6_pcIM(mM,:);
	pred_panel_shift_6_pcTM     = pred_panel_6_pcTM(mM,:);
	pred_panel_shift_6_pcIMdiff = pred_panel_6_pcIMdiff(mM,:);
	pred_panel_shift_6_pcTMdiff = pred_panel_6_pcTMdiff(mM,:);
	pred_panel_shift_7_pcIM     = pred_panel_7_pcIM(mM,:);
	pred_panel_shift_7_pcTM     = pred_panel_7_pcTM(mM,:);
	pred_panel_shift_7_pcIMdiff = pred_panel_7_pcIMdiff(mM,:);
	pred_panel_shift_7_pcTMdiff = pred_panel_7_pcTMdiff(mM,:);
	pred_panel_shift_8_pcIM     = pred_panel_8_pcIM(mM,:);
	pred_panel_shift_8_pcTM     = pred_panel_8_pcTM(mM,:);
	pred_panel_shift_8_pcIMdiff = pred_panel_8_pcIMdiff(mM,:);
	pred_panel_shift_8_pcTMdiff = pred_panel_8_pcTMdiff(mM,:);
	pred_panel_shift_9_pcIM     = pred_panel_9_pcIM(mM,:);
	pred_panel_shift_9_pcTM     = pred_panel_9_pcTM(mM,:);
	pred_panel_shift_9_pcIMdiff = pred_panel_9_pcIMdiff(mM,:);
	pred_panel_shift_9_pcTMdiff = pred_panel_9_pcTMdiff(mM,:);
	pred_panel_shift_10_pcIM     = pred_panel_10_pcIM(mM,:);
	pred_panel_shift_10_pcTM     = pred_panel_10_pcTM(mM,:);
	pred_panel_shift_10_pcIMdiff = pred_panel_10_pcIMdiff(mM,:);
	pred_panel_shift_10_pcTMdiff = pred_panel_10_pcTMdiff(mM,:);
	pred_panel_shift_11_pcIM     = pred_panel_11_pcIM(mM,:);
	pred_panel_shift_11_pcTM     = pred_panel_11_pcTM(mM,:);
	pred_panel_shift_11_pcIMdiff = pred_panel_11_pcIMdiff(mM,:);
	pred_panel_shift_11_pcTMdiff = pred_panel_11_pcTMdiff(mM,:);
	pred_panel_shift_12_pcIM     = pred_panel_12_pcIM(mM,:);
	pred_panel_shift_12_pcTM     = pred_panel_12_pcTM(mM,:);
	pred_panel_shift_12_pcIMdiff = pred_panel_12_pcIMdiff(mM,:);
	pred_panel_shift_12_pcTMdiff = pred_panel_12_pcTMdiff(mM,:);
	pred_panel_shift_13_pcIM     = pred_panel_13_pcIM(mM,:);
	pred_panel_shift_13_pcTM     = pred_panel_13_pcTM(mM,:);
	pred_panel_shift_13_pcIMdiff = pred_panel_13_pcIMdiff(mM,:);
	pred_panel_shift_13_pcTMdiff = pred_panel_13_pcTMdiff(mM,:);
	pred_panel_shift_14_pcIM     = pred_panel_14_pcIM(mM,:);
	pred_panel_shift_14_pcTM     = pred_panel_14_pcTM(mM,:);
	pred_panel_shift_14_pcIMdiff = pred_panel_14_pcIMdiff(mM,:);
	pred_panel_shift_14_pcTMdiff = pred_panel_14_pcTMdiff(mM,:);
	pred_panel_shift_15_pcIM     = pred_panel_15_pcIM(mM,:);
	pred_panel_shift_15_pcTM     = pred_panel_15_pcTM(mM,:);
	pred_panel_shift_15_pcIMdiff = pred_panel_15_pcIMdiff(mM,:);
	pred_panel_shift_15_pcTMdiff = pred_panel_15_pcTMdiff(mM,:);
	pred_panel_shift_16_pcIM     = pred_panel_16_pcIM(mM,:);
	pred_panel_shift_16_pcTM     = pred_panel_16_pcTM(mM,:);
	pred_panel_shift_16_pcIMdiff = pred_panel_16_pcIMdiff(mM,:);
	pred_panel_shift_16_pcTMdiff = pred_panel_16_pcTMdiff(mM,:);

	S = fullfile(strcat(compDataDir,'comp_data.mat'));
	save(S);

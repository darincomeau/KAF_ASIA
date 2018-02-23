%% sea ice volume anomaly experiments

% clear all
close all

% region = 'scratch'
% region = 'Arctic';
% region = 'ChukchiBeaufort';
% region = 'Beaufort';
% region = 'Chukchi';
% region = 'EastSibLaptev';
% region = 'EastSib';
% region = 'Laptev';
% region = 'BarentsKara';
% region = 'Barents';
% region = 'Kara';
% region = 'Greenland';
% region = 'Baffin';
% region = 'Labrador';
% region = 'Hudson';
% region = 'Bering';
% region = 'Okhotsk';

% embedWin = 12;

% varsUsed = 'SIC';
% varsUsed = 'SIC_SST';
% varsUsed = 'SIC_SST_SIT';
% varsUsed = 'SIC_SST_SIT_SLP';

% predOn = 1;
% fullDataOn = 1;

%% define experiment
switch region

    case 'scratch'
        lonLim = [165 200];
        latLim = [55 65];

    case 'Arctic'
        lonLim = [0 360];
        latLim = [45 90];

    case 'ChukchiBeaufort'
        lonLim = [175 235];
        latLim = [65 75];

    case 'Chukchi'
        lonLim = [175 205];
        latLim = [65 75];

    case 'Beaufort'
        lonLim = [205 235];
        latLim = [65 75];

    case 'EastSibLaptev'
        lonLim = [105 175];
        latLim = [65 75];

    case 'EastSib'
        lonLim = [140 175];
        latLim = [65 75];

    case 'Laptev'
        lonLim = [105 140];
        latLim = [70 80];

    case 'BarentsKara'
        lonLim = [30 95];
        latLim = [65 80];

    case 'Barents'
        lonLim = [30 60];
        latLim = [65 80];

    case 'Kara'
        lonLim = [60 95];
        latLim = [65 80];

    case 'Greenland'
        lonLim = [325 360];
        latLim = [65 80];

    case 'Baffin'
        lonLim = [280 310];
        latLim = [70 80];

    case 'Labrador'
        lonLim = [290 310];
        latLim = [50 70];

    case 'Hudson'
        lonLim = [265 285];
        latLim = [55 65];

    case 'Bering'
        lonLim = [165 200];
        latLim = [55 65];

    case 'Okhotsk'
        lonLim = [135 165];
        latLim = [45 65];

    case 'CentralArctic'
        lonLim = [0 360];
        latLim = [70 90];        

end

switch varsUsed

    case 'SIC'
        varsFlag = 1;

    case 'SIC_SST'
        varsFlag = 2;

    case 'SIC_SST_SIT'
        varsFlag = 3;

    case 'SIC_SST_SIT_SLP'
        varsFlag = 4;

    case 'SIC_SIT'
        varsFlag = 5;

    case 'SIC_SLP'
        varsFlag = 6;


    case 'SIC_SST_SLP'
        varsFlag = 7;

    case 'SIC_SST_SITq48'
        varsFlag = 8;               

end


saveTag = strcat(region,'_',varsUsed,'_q',num2str(embedWin),'_train_',num2str(trainLim(1)),'_',num2str(trainLim(2)));
saveDir = fullfile('output/',saveTag,'/');
predDir = fullfile('output/predictions/',saveTag,'/');
arcticTag = strcat('Arctic_',varsUsed,'_q',num2str(embedWin),'_train_',num2str(trainLim(1)),'_',num2str(trainLim(2)));
arcticDir = fullfile('output/',arcticTag,'/');

if exist(saveDir) == 0
    mkdir(saveDir)
end

if exist(predDir) == 0
    mkdir(predDir)
end

trainTag = sprintf( 'x%i-%i_y%i-%i_yr%i-%i', lonLim( 1 ), ...
                                             lonLim( 2 ), ...
                                             latLim( 1 ), ...
                                             latLim( 2 ), ...
                                             trainLim( 1 ), ...
                                             trainLim( 2 ) );

testTag = sprintf( 'x%i-%i_y%i-%i_yr%i-%i', lonLim( 1 ), ...
                                            lonLim( 2 ), ...
                                            latLim( 1 ), ...
                                            latLim( 2 ), ...
                                            testLim( 1 ), ...
                                            testLim( 2 ) );

%% get raw data
checkDir = fullfile('data/raw/CCSM4_Data/b40.1850/SIC/',trainTag,'/data/');
if exist(checkDir) == 0  
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),trainLim(1),trainLim(2),'SIC')
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),testLim(1),testLim(2),'SIC')
end

checkDir = fullfile('data/raw/CCSM4_Data/b40.1850/SST/',trainTag,'/data/');
if exist(checkDir) == 0
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),trainLim(1),trainLim(2),'SST')
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),testLim(1),testLim(2),'SST')
end

checkDir = fullfile('data/raw/CCSM4_Data/b40.1850/SIT/',trainTag,'/data/');
if exist(checkDir) == 0
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),trainLim(1),trainLim(2),'SIT')
    ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),testLim(1),testLim(2),'SIT')
end

%% calculate anomalies
checkFile = fullfile(strcat(saveDir,'IVA.mat'));

if exist(checkFile) == 0

    dataTrainPath_area = fullfile('data/raw/CCSM4_Data/b40.1850/SIA/',trainTag,'/data/');
    dataTestPath_area = fullfile('data/raw/CCSM4_Data/b40.1850/SIA/',testTag,'/data/');
    if exist(dataTrainPath_area) == 0
        ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),trainLim(1),trainLim(2),'SIA')
        ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),testLim(1),testLim(2),'SIA')
    end

    dataTrainPath_thick = fullfile('data/raw/CCSM4_Data/b40.1850/SIT_UW/',trainTag,'/data/');
    dataTestPath_thick = fullfile('data/raw/CCSM4_Data/b40.1850/SIT_UW/',testTag,'/data/');
    if exist(dataTrainPath_thick) == 0
        ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),trainLim(1),trainLim(2),'SIT_UW')
        ccsm4DataRaw(lonLim(1),lonLim(2),latLim(1),latLim(2),testLim(1),testLim(2),'SIT_UW')
    end

    load ( fullfile( dataTrainPath_area, 'dataX.mat' ) )
    ice_area = x; % dimensions space x time    
    load ( fullfile( dataTrainPath_thick, 'dataX.mat' ) )
    ice_thick = x; % dimensions space x time 
    ice_raw = ice_area.*ice_thick;

    % domain climatology
    ice_clim = zeros(12,1);
    ice_anom_train = zeros(nT,1);
    for i = 1:12
        month = 0;
        j = 1;
        while i + (j-1)*12 <=nT
            month(j) = sum(ice_raw(:,i+(j-1)*12));
            j = j+1;
        end
        ice_clim(i) = mean(month);
    end
    for i = 1:nT
        if mod(i,12) ~= 0
            ice_anom_train(i) = sum(ice_raw(:,i)) - ice_clim(mod(i,12));
        elseif mod(i,12) == 0
            ice_anom_train(i) = sum(ice_raw(:,i)) - ice_clim(12);
        end
    end

    load ( fullfile( dataTestPath_area, 'dataX.mat' ) )
    ice_area = x; % dimensions space x time
    load ( fullfile( dataTestPath_thick, 'dataX.mat' ) )
    ice_thick = x; % dimensions space x time
    ice_raw = ice_area.*ice_thick;

    ice_anom_test = zeros(nT,1);
    for i = 1:nT
        if mod(i,12) ~= 0
            ice_anom_test(i) = sum(ice_raw(:,i)) - ice_clim(mod(i,12));
        elseif mod(i,12) == 0
            ice_anom_test(i) = sum(ice_raw(:,i)) - ice_clim(12);
        end
    end

    ice_anom_train = ice_anom_train*10^(-9);
    ice_anom_test  = ice_anom_test*10^(-9);
    ice_clim       = ice_clim*10^(-9);

    S = fullfile(strcat(saveDir,'IVA.mat'));
    save(S,'ice_anom_train','ice_anom_test','ice_clim')

end

%% evaluate kernel distances, either based on region only or pan-Arctic data
if fullDataOn == 0
    checkFile = fullfile(strcat(saveDir,'pMatrix.mat'));
end
if fullDataOn == 1
    checkFile = fullfile(strcat(arcticDir, 'pMatrix.mat'))
end

if exist(checkFile) == 0

    if fullDataOn == 0
        model = nlsa_ose_ica( lonLim, latLim, trainLim, testLim, embedWin, varsFlag );
    end
    if fullDataOn == 1
        model = nlsa_ose_ica( [0 360], [45 90], trainLim, testLim, embedWin, varsFlag );
    end
    disp( 'Embedding' ); makeEmbedding( model )
    disp( 'Phase space velocity' ); computePhaseSpaceVelocity( model, 'ifWriteXi', true )
    fprintf( 'Pairwise distances, %i/%i\n', 1, 1 ); computePairwiseDistances( model, 1, 1 )
    disp( 'OSE embedding' ); makeOSEEmbedding( model );
    disp( 'OSE phase space velocity' ); computeOSEPhaseSpaceVelocity( model, 'ifWriteXi', true )
    fprintf( 'OSE pairwise distances, %i/%i\n', 1, 1 ); computeOSEPairwiseDistances( model, 1, 1 )

    distPath = model.pDistance.path;
    distPath = fullfile( distPath, model.pDistance.pathY );

    % dcnote find cleaner way
    if embedWin == 6
        load ( fullfile( distPath, 'dataY_1_1-4791.mat' ) );
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd;

    elseif embedWin == 12
        % load ( fullfile( distPath, 'dataY_1_1-4785.mat' ) );
        load ( fullfile( distPath, 'dataY_1_1-4749.mat' ) ); 
        % load ( fullfile( distPath, 'dataY_1_1-465.mat' ) );           
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd; 

    elseif embedWin == 24
        load ( fullfile( distPath, 'dataY_1_1-4773.mat' ) );
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd;

    end

    dist_ord = zeros(size(p1));
    [disti, distj] = size(p1);
    for i = 1:disti
        for j = 1:distj
            dist_ord(i,y1(i,j)) = p1(i,j);
        end
    end

    clear p1 y1

    oseDistPath = model.osePDistance.path;
    oseDistPath = fullfile( oseDistPath, model.osePDistance.pathY );

    if embedWin == 6
        load ( fullfile( oseDistPath, 'dataY_1_1-4791.mat' ) );
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd;

    elseif embedWin == 12
        % load ( fullfile( oseDistPath, 'dataY_1_1-4785.mat' ) );
        load ( fullfile( oseDistPath, 'dataY_1_1-4749.mat' ) );
        % load ( fullfile( oseDistPath, 'dataY_1_1-465.mat' ) );         
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd;

    elseif embedWin == 24
        load ( fullfile( oseDistPath, 'dataY_1_1-4773.mat' ) );
        p1 = yVal;
        y1 = yInd;
        clear yVal yInd;

    end

    % form kernel matrix

    oseDist_ord = zeros(size(p1));
    [disti, distj] = size(p1);
    for i = 1:disti
        for j = 1:distj
            oseDist_ord(i,y1(i,j)) = p1(i,j);
        end
    end

    clear p1 y1

    if fullDataOn == 0
        S = fullfile(strcat(saveDir,'pMatrix.mat'));
    end

    if fullDataOn == 1
        S = fullfile(strcat(arcticDir,'pMatrix.mat'));
    end

    save(S,'dist_ord','oseDist_ord')

    rmdir('ica_scratch/*','s')

end

%% form predictions
checkFile = fullfile(strcat(saveDir,'pred_iva',num2str(fullDataOn),'.mat'));

if exist(checkFile) == 0 | predOn == 1

    S = fullfile(strcat(saveDir,'IVA.mat'));
    load(S)
    % note shift by embedding window, as well as adjustment for phase space
    % velocity calculation (In.fdOrder = 4 in nlsa_ose_ica.m)
    data_train = ice_anom_train(embedWin + 2: end - 2);
    data_test = ice_anom_test(embedWin + 2: end - 2);

    if fullDataOn == 0
        S = fullfile(strcat(saveDir,'pMatrix.mat'));
    end
    if fullDataOn == 1
        S = fullfile(strcat(arcticDir,'pMatrix.mat'));
    end    
    load(S)

    % Laplacian pyramid extention inputs
    tLag = 13;
    d_ref = dist_ord(1:end-tLag,1:end-tLag);
    d_ose = oseDist_ord(:,1:end-tLag);
    f_ref = data_train;
    A=d_ref;
    A(d_ref==0)=NaN;
    eps_0 = nanmedian(nanmedian(A));
    % eps_0 = .1*eps_0;

    decomp = LP_precomp(d_ref, f_ref, tLag, eps_0);
    V_pred = LP_forecast(d_ose, eps_0, decomp);

    f_ext = V_pred(:,1);

    % check for NaNs in predictions - make loop over loosening tol_cons
    S = fullfile(strcat(saveDir,'nan_sum.mat'));
    if exist(S) > 0
        delete(S)
    end
    if sum(sum(isnan(V_pred))) > 0
        nan_sum = sum(sum(isnan(V_pred)))
        save(S,'nan_sum')
    end

    % errors for all
    pred_traj = V_pred(1:end-tLag+1,:);

    % format truth
    [nIter, tmp] = size(pred_traj);
    truth = zeros(nIter,tLag);
    for j = 1:nIter
        for i = 1:tLag
            truth(j,i) = data_test(j+i-1);
        end
    end

    % format damped persistence
    % calculcate AR1 coefficient for damped persistence
    [acf,lags] = autocorr(data_test,1);
    ar1coef = acf(2)
    pred_trajDP = zeros(nIter,tLag);
    for j = 1:nIter
        for i = 1:tLag
            pred_trajDP(j,i) = ar1coef^(i-1)*truth(j,1);
        end
    end

    [pred_rms, pred_pc, pred_rmsP, pred_pcP, pValues, std_truth] = calc_errors_v2(pred_traj, truth);
    [pred_rmsDP, pred_pcDP, pred_rmsP, pred_pcP, pValuesDP, std_tmp] = calc_errors_v2(pred_trajDP, truth);

    % condition trajectories on initial month
    pred_rmsIM  = zeros(12,tLag);
    pred_pcIM   = zeros(12,tLag);
    pred_rmsIMP = zeros(12,tLag);
    pred_pcIMP  = zeros(12,tLag);
    pred_rmsIMDP = zeros(12,tLag);
    pred_pcIMDP  = zeros(12,tLag);

    pred_rmsTM  = zeros(12,tLag);
    pred_pcTM   = zeros(12,tLag);
    pred_rmsTMP = zeros(12,tLag);
    pred_pcTMP  = zeros(12,tLag);
    pred_rmsTMDP = zeros(12,tLag);
    pred_pcTMDP  = zeros(12,tLag);

    mIter = floor(nIter/12);

    for m = 1:12
        initM = m;

        % initial month - predictions start in month initM
        pred_trajM = zeros(mIter,tLag);
        pred_trajDPM = zeros(mIter,tLag);
        truthM = zeros(mIter,tLag);

        for j = 1:mIter-1
            pred_trajM(j,:) = pred_traj(initM + (j-1)*12, :);
            pred_trajDPM(j,:) = pred_trajDP(initM + (j-1)*12, :);
            truthM(j,:) = truth(initM + (j-1)*12, :);
        end

        [pred_rms_tmp, pred_pc_tmp, pred_rmsP_tmp, pred_pcP_tmp, pValuesIM, std_truthIM] = calc_errors_v2(pred_trajM, truthM);
        [pred_rmsDP_tmp, pred_pcDP_tmp, pred_rmsP_tmp, pred_pcP_tmp, pValuesIMDP, std_tmp] = calc_errors_v2(pred_trajDPM, truthM);

        pred_rmsIM(initM,:)  = pred_rms_tmp;
        pred_pcIM(initM,:)   = pred_pc_tmp;
        pred_rmsIMP(initM,:) = pred_rmsP_tmp;
        pred_pcIMP(initM,:)  = pred_pcP_tmp;
        pred_rmsIMDP(initM,:) = pred_rmsDP_tmp;
        pred_pcIMDP(initM,:)  = pred_pcDP_tmp;

        % target month - predictions end in month initM
        pred_trajM = zeros(mIter,tLag);
        pred_trajDPM = zeros(mIter,tLag);
        truthM = zeros(mIter,tLag);

        for j = 1:mIter-1
            for i = 1:tLag
                pred_trajM(j,i) = pred_traj(initM + (i-1) + (j-1)*12, i);
                pred_trajDPM(j,i) = pred_trajDP(initM + (i-1) + (j-1)*12, i);
                truthM    (j,i) = truth    (initM + (i-1) + (j-1)*12, i);
            end
        end

        [pred_rms_tmp, pred_pc_tmp, pred_rmsP_tmp, pred_pcP_tmp, pValuesTM, std_truthTM] = calc_errors_v2(pred_trajM, truthM);
        [pred_rmsDP_tmp, pred_pcDP_tmp, pred_rmsP_tmp, pred_pcP_tmp, pValuesTMDP, std_tmp] = calc_errors_v2(pred_trajDPM, truthM);

        pred_rmsTM(initM,:)  = pred_rms_tmp;
        pred_pcTM(initM,:)   = pred_pc_tmp;
        pred_rmsTMP(initM,:) = pred_rmsP_tmp;
        pred_pcTMP(initM,:)  = pred_pcP_tmp;
        pred_rmsTMDP(initM,:) = pred_rmsDP_tmp;
        pred_pcTMDP(initM,:)  = pred_pcDP_tmp;

    end

    % S = fullfile(strcat(saveDir,'pred_ica.mat'));
    S = fullfile(strcat(predDir,'pred_iva',num2str(fullDataOn),'.mat'));
    save(S,'pred_traj','pred_rms','pred_pc','pred_rmsIM','pred_pcIM','pred_rmsTM','pred_pcTM', ...
        'pred_rmsP','pred_pcP','pred_rmsIMP','pred_pcIMP','pred_rmsTMP','pred_pcTMP', ...
        'pred_rmsDP','pred_pcDP','pred_rmsIMDP','pred_pcIMDP','pred_rmsTMDP','pred_pcTMDP', 'ar1coef', ...
        'f_ext','truth','d_ref','d_ose','tLag','data_train','data_test','eps_0', 'pred_trajDP', 'pValues', 'pValuesTM', ...
        'std_truth', 'std_truthIM', 'std_truthTM')

end

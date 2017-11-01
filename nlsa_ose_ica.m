function model = nlsa_ose_ica( lonLim, latLim, trainLim, testLim, embedWin, varsFlag )


        % In-sample dataset parameters ("nature")
        In.yrLim          = [trainLim(1) trainLim(2) ];
        In.tFormat        = 'yyyymm';
        In.experiment     = 'CCSM4_Data/b40.1850';

        if varsFlag == 1
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 2
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SST';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 3
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SST';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;

            In.Src( 3 ).field = 'SIT';
            In.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 3 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 4
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SST';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;

            In.Src( 3 ).field = 'SIT';
            In.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 3 ).idxE  = 1 : embedWin;

            In.Src( 4 ).field = 'SLP_train';
            In.Src( 4 ).xLim  = [ 0 360 ];
            In.Src( 4 ).yLim  = [ 45  90 ];
            In.Src( 4 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 5
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SIT';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 6
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SLP_train';
            In.Src( 2 ).xLim  = [ 0 360 ];
            In.Src( 2 ).yLim  = [ 45 90 ];
            In.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 7
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SST';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;

            In.Src( 3 ).field = 'SLP_train';
            In.Src( 3 ).xLim  = [ 0 360 ];
            In.Src( 3 ).yLim  = [ 45  90 ];
            In.Src( 3 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 8
            In.Src( 1 ).field = 'SIC';
            In.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 1 ).idxE  = 1 : embedWin;

            In.Src( 2 ).field = 'SST';
            In.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 2 ).idxE  = 1 : embedWin;

            In.Src( 3 ).field = 'SIT';
            In.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            In.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            In.Src( 3 ).idxE  = 1 : 48;
        end                

        In.trgExperiment  = In.experiment;
        In.Trg            = In.Src;

        % Out-of-sample dataset parameters ("model")
        Out.yrLim          = [ testLim(1) testLim(2) ];
        Out.tFormat        = 'yyyymm';
        Out.experiment     = 'CCSM4_Data/b40.1850';
 
        if varsFlag == 1
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 2
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SST';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 3
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SST';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;

            Out.Src( 3 ).field = 'SIT';
            Out.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 3 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 4
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SST';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;

            Out.Src( 3 ).field = 'SIT';
            Out.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 3 ).idxE  = 1 : embedWin;

            Out.Src( 4 ).field = 'SLP_test';
            Out.Src( 4 ).xLim  = [ 0 360 ];
            Out.Src( 4 ).yLim  = [ 45  90 ];
            Out.Src( 4 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 5
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SIT';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 6
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SLP_test';
            Out.Src( 2 ).xLim  = [ 0 360 ];
            Out.Src( 2 ).yLim  = [ 45 90 ];
            Out.Src( 2 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 7
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SST';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;

            Out.Src( 3 ).field = 'SLP_test';
            Out.Src( 3 ).xLim  = [ 0 360 ];
            Out.Src( 3 ).yLim  = [ 45  90 ];
            Out.Src( 3 ).idxE  = 1 : embedWin;
        end

        if varsFlag == 8
            Out.Src( 1 ).field = 'SIC';
            Out.Src( 1 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 1 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 1 ).idxE  = 1 : embedWin;

            Out.Src( 2 ).field = 'SST';
            Out.Src( 2 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 2 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 2 ).idxE  = 1 : embedWin;

            Out.Src( 3 ).field = 'SIT';
            Out.Src( 3 ).xLim  = [ lonLim(1) lonLim(2) ];
            Out.Src( 3 ).yLim  = [ latLim(1) latLim(2) ];
            Out.Src( 3 ).idxE  = 1 : 48;
        end              

        Out.trgExperiment  = Out.experiment;
        Out.Trg            = Out.Src;

        In.nN           = 100;      % nearest neighbors for pairwise distances
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance

        Out.nN           = 100;      % nearest neighbors for pairwise distances
        Out.nNS          = Out.nN;    % nearest neighbors for symmetric distance

        % NLSA parameters (in-sample)
        In.nXB          = 2;
        In.nXA          = 2;
        In.fdOrder      = 4;
        In.fdType       = 'central';
        In.embFormat    = 'evector'; % storage format for source data
        In.embFormatTrg = In.embFormat; % storage format for target data
        In.nB           = 1;         % bathches to partition the source data
        In.nBTrg        = 1;         % batches to partition the target data
        In.tol          = 0;         % tolerance (for generalized eigenvalue problem weights only)
        In.zeta         = 0.995;     % for cone kernels
        In.distNorm     = 'geometric'; % kernel normalization (geometric/harmonic)
%         In.nN           = 4773;      % nearest neighbors for pairwise distances
%         In.nN           = 9573;      % nearest neighbors for pairwise distances
%         In.nNS          = In.nN;        % nearest neighbors for symmetric distance
%         In.epsilon      = 1;        % Gaussian Kernel width 
        In.epsilon      = 2;        % Gaussian Kernel width 
        In.alpha        = 0;         % Kernel normalization 
        In.nPhi         = 50;        % Laplace-Beltrami eigenfunctions
        In.idxPhi       = 1 : In.nPhi;  % eigenfunctions used for linear mapping
        In.nProc        = 1;         % number of processes for distance matrix
        In.iProc        = 1;         % process of the current script

        % NLSA parameters (out-of-sample)
        Out.nXB          = 2;
        Out.nXA          = 2;
        Out.fdOrder      = 4;
        Out.fdType       = 'central';
        Out.embFormat    = 'evector'; % storage format for source data
        Out.embFormatTrg = Out.embFormat; % storage format for target data
        Out.nB           = 1;         % bathches to partition the source data
        Out.nBTrg        = 1;         % batches to partition the target data
        Out.tol          = 1E-7;      % tolerance (for generalized eigenvalue problem weights only)
        Out.zeta         = 0.995;     % for cone kernels
        Out.distNorm     = 'geometric'; % kernel normalization (geometric/harmonic)
%         Out.nN           = 4773;      % nearest neighbors for pairwise distances
%        Out.nN           = 9573;      % nearest neighbors for pairwise distances
%         Out.nNS          = Out.nN;        % nearest neighbors for symmetric distance
%         Out.epsilon      = 10;        % Gaussian Kernel width 
%         Out.alpha        = 1;         % Kernel normalization 
        Out.epsilon      = 2;        % Gaussian Kernel width 
        Out.alpha        = 0;         % Kernel normalization 
        Out.nPhi         = 50;        % Laplace-Beltrami eigenfunctions
%         Out.epsilonOSE   = 1;
        Out.epsilonOSE   = 2;
        Out.alphaOSE     = 0;
        Out.nPhiOSE      = 50;
        Out.idxPhi       = 1 : Out.nPhi;  % eigenfunctions used for linear mapping
        Out.nProc        = 1;         % number of processes for distance matrix
        Out.iProc        = 1;         % process of the current script


        % NLSA kernel
%        pDist = nlsaPairwiseDistance_cone( 'nearestNeighbors', In.nN, ...
%                                           'tolerance', In.tol, ...
%                                           'zeta', In.zeta, ...
%                                           'normalization', In.distNorm );
% 
%        osePDist = nlsaPairwiseDistance_cone( 'nearestNeighbors', Out.nN, ...
%                                              'tolerance', Out.tol, ...
%                                              'zeta', Out.zeta, ...
%                                              'normalization', Out.distNorm );

         pDist = nlsaPairwiseDistance_at( 'nearestNeighbors', In.nN, ...
                                          'normalization', 'geometric' );
 
         osePDist = nlsaPairwiseDistance_at( 'nearestNeighbors', Out.nN, ...
                                             'normalization', 'geometric' );

%% NLSA MODEL
In.nYr  = In.yrLim( 2 ) - In.yrLim( 1 ) + 1;
In.nS   = 12 * In.nYr; % sample number
In.tNum = zeros( 1,  In.nS );
iS   = 1;
for iYr = 1 : In.nYr
    for iM = 1 : 12
        In.tNum( iS ) = datenum( sprintf( '%04i%02i', In.yrLim( 1 ) + iYr - 1, iM  ), 'yyyymm' );
        iS         = iS + 1;
    end
end
In.nS      = numel( In.tNum );
In.nC      = numel( In.Src );  % number of source components
In.nCT   = numel( In.Trg );  % number of target compoents
In.nE      = In.Src( 1 ).idxE( end );
for iC = 2 : In.nC
    In.nE = max( In.nE, In.Src( iC ).idxE( end ) );
end
In.nET  = In.Trg( 1 ).idxE( end );
for iC = 2 : In.nCT
    In.nET = max( In.nET, In.Trg( iC ).idxE( end ) );
end
In.idxT1   = max( In.nE, In.nET ) + In.nXB;               % time origin for embedding
In.nSE     = In.nS - In.idxT1 + 1 - In.nXA;                     % sample number after embedding

Out.nYr  = Out.yrLim( 2 ) - Out.yrLim( 1 ) + 1;
Out.nS   = 12 * Out.nYr; % sample number
Out.tNum = zeros( 1,  Out.nS );
iS   = 1;
for iYr = 1 : Out.nYr
    for iM = 1 : 12
        Out.tNum( iS ) = datenum( sprintf( '%04i%02i', Out.yrLim( 1 ) + iYr - 1, iM  ), 'yyyymm' );
        iS         = iS + 1;
    end
end
Out.nS      = numel( Out.tNum );
Out.nC      = numel( Out.Src );  % number of source components
Out.nCT     = numel( Out.Trg );  % number of target compoents
Out.nE      = Out.Src( 1 ).idxE( end );
for iC = 2 : Out.nC
    Out.nE = max( Out.nE, Out.Src( iC ).idxE( end ) );
end
Out.nET  = Out.Trg( 1 ).idxE( end );
Out.idxT1   = max( Out.nE, Out.nET ) + Out.nXB;               % time origin for embedding
Out.nSE     = Out.nS - Out.idxT1 + 1 - Out.nXA;                     % sample number after embedding


nlsaPath = fullfile('/kontiki_array4/comeau/nlsa/nlsa_v2014/examples/KAF_ASIA/ica_scratch' );

% Source data
for iC = In.nC : -1 : 1

    xyr = sprintf( 'x%i-%i_y%i-%i_yr%i-%i', In.Src( iC ).xLim( 1 ), ...
                                           In.Src( iC ).xLim( 2 ), ...
                                           In.Src( iC ).yLim( 1 ), ...
                                           In.Src( iC ).yLim( 2 ), ...
                                           In.yrLim( 1 ), ...
                                           In.yrLim( 2 ) );
%dcmod    
%     path = fullfile( pwd,  ...
%                      'data/raw', ...
%                      In.experiment, ...
%                      In.Src( iC ).field, ...
%                      xyr );

    path = fullfile('./data/raw/', ...
                     In.experiment, ...
                     In.Src( iC ).field, ...
                     xyr, ...
                     '/data' );
                             
    tag = [ In.experiment '_' In.Src( iC ).field '_' xyr ];

    load( fullfile( path, 'dataGrid.mat' ), 'nD' ); % source data dimension

    partition    = nlsaPartition( 'nSample', In.nS ); % source data assumed to be stored in a single batch


    srcComponent( iC ) = nlsaComponent( 'partition', partition, ...
                                        'dimension', nD, ...
                                        'path',      path, ...
                                        'file',      'dataX.mat', ...
                                        'tag',       tag );

    embComponent( iC )    = nlsaEmbeddedComponent_xi( 'idxE', In.Src( iC ).idxE, ...
                                                      'nXB', In.nXB, ... 
                                                      'nXA', In.nXA, ...
                                                      'fdOrder',  In.fdOrder, ...
                                                      'fdType', In.fdType, ...
                                                      'storageFormat', In.embFormat );
end

% Target data
for iCT = In.nCT : -1 : 1

    xyr = sprintf( 'x%i-%i_y%i-%i_yr%i-%i', In.Trg( iCT ).xLim( 1 ), ...
                                            In.Trg( iCT ).xLim( 2 ), ...
                                            In.Trg( iCT ).yLim( 1 ), ...
                                            In.Trg( iCT ).yLim( 2 ), ...
                                            In.yrLim( 1 ), ...
                                            In.yrLim( 2 ) );
                 
    path = fullfile('./data/raw/', ...
                     In.experiment, ...
                     In.Src( iCT ).field, ...
                     xyr, ...
                     '/data' );          
                                                   
    tag = [ In.experiment '_' In.Trg( iCT ).field '_' xyr ];

    load( fullfile( path, 'dataGrid.mat' ), 'nD' ); % source data dimension

    partition    = nlsaPartition( 'nSample', In.nS ); % source data assumed to be stored in a single batch


    trgComponent( iCT ) = nlsaComponent( 'partition', partition, ...
                                        'dimension', nD, ...
                                        'path',      path, ...
                                        'file',      'dataX.mat', ...
                                        'tag',       tag );

    trgEmbComponent( iCT )    = nlsaEmbeddedComponent_xi( 'idxE', In.Trg( iCT ).idxE, ...
                                                         'nXB', In.nXB, ...
                                                         'nXA', In.nXA, ...
                                                         'fdOrder',  In.fdOrder, ...
                                                         'fdType', In.fdType, ...
                                                         'storageFormat', In.embFormat );

end


% OSE data
for iC = Out.nC : -1 : 1

    xyr = sprintf( 'x%i-%i_y%i-%i_yr%i-%i', Out.Src( iC ).xLim( 1 ), ...
                                            Out.Src( iC ).xLim( 2 ), ...
                                            Out.Src( iC ).yLim( 1 ), ...
                                            Out.Src( iC ).yLim( 2 ), ...
                                            Out.yrLim( 1 ), ...
                                            Out.yrLim( 2 ) );

    path = fullfile('./data/raw/', ...
                     Out.experiment, ...
                     Out.Src( iC ).field, ...
                     xyr, ...
                     '/data' );             
                 
    tag = [ Out.experiment '_' Out.Src( iC ).field '_' xyr ];

    load( fullfile( path, 'dataGrid.mat' ), 'nD' ); % source data dimension

    partition    = nlsaPartition( 'nSample', Out.nS ); % source data assumed to be stored in a single batch


    oseComponent( iC ) = nlsaComponent( 'partition', partition, ...
                                        'dimension', nD, ...
                                        'path',      path, ...
                                        'file',      'dataX.mat', ...
                                        'tag',       tag );

    oseEmbComponent( iC ) = nlsaEmbeddedComponent_xi( 'idxE', Out.Src( iC ).idxE, ...
                                                      'nXB', Out.nXB, ... 
                                                      'nXA', Out.nXA, ...
                                                      'fdOrder',  Out.fdOrder, ...
                                                      'fdType', Out.fdType, ...
                                                      'storageFormat', Out.embFormat );
end

% %dcmod add
srcComponent = srcComponent';
trgComponent = trgComponent';
oseComponent = oseComponent';

embComponent = embComponent';
trgEmbComponent = trgEmbComponent';
oseEmbComponent = oseEmbComponent';

embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch', In.nB );
trgEmbPartition = nlsaPartition( 'nSample', In.nSE', 'nBatch', In.nBTrg );
oseEmbPartition = nlsaPartition( 'nSample', Out.nSE', 'nBatch', Out.nBTrg );

sDist = nlsaSymmetricDistance( 'nearestNeighbors', In.nNS ); % symmetrized kernel

diffOp = nlsaDiffusionOperator( 'alpha',   In.alpha, ...
                                'epsilon', In.epsilon, ...
                                'nEigenfunction', In.nPhi ); % Laplace-Beltrami operator

oseDiffOp = nlsaDiffusionOperator_ose( 'alphaOSE', Out.alphaOSE, ...
                                       'epsilonOSE', Out.epsilonOSE, ...
                                       'nEigenfunctionOSE', Out.nPhiOSE );

entDiffOp = nlsaDiffusionOperator( 'alpha', Out.alpha, ...
                                   'epsilon', Out.epsilon, ...
                                   'nEigenfunction', Out.nPhi );

% Array of linear maps with nested sets of basis functions
for iL = numel( In.idxPhi ) : -1 : 1
    linMap( iL ) = nlsaLinearMap_basis( 'basisFunctionIdx', In.idxPhi( 1 : iL ) );
end

% Build NLSA model    
model = nlsaModel_ent( 'path',                             nlsaPath, ...
                       'srcTime',                          In.tNum, ...
                       'timeFormat',                       In.tFormat, ...
                       'sourceComponent',                  srcComponent, ...
                       'embeddingOrigin',                  In.idxT1, ...
                       'embeddingTemplate',                embComponent, ...
                       'partitionTemplate',                embPartition, ...
                       'pairwiseDistanceTemplate',         pDist, ...
                       'symmetricDistanceTemplate',        sDist, ...
                       'diffusionOperatorTemplate',        diffOp, ...
                       'targetComponent',                  trgComponent, ...
                       'targetEmbeddingOrigin',            In.idxT1, ...
                       'targetEmbeddingTemplate',          trgEmbComponent, ...
                       'targetPartitionTemplate',          trgEmbPartition, ...
                       'linearMapTemplate',                linMap, ...
                       'oseComponent',                     oseComponent, ...
                       'oseTime',                          Out.tNum, ...
                       'oseTimeFormat',                    Out.tFormat, ...
                       'oseEmbeddingOrigin',               Out.idxT1, ...
                       'oseEmbeddingTemplate',             oseEmbComponent, ...
                       'osePartitionTemplate',             oseEmbPartition, ...  
                       'osePairwiseDistanceTemplate',      osePDist, ...
                       'oseDiffusionOperatorTemplate',     oseDiffOp, ...
                       'entropyDiffusionOperatorTemplate', entDiffOp );

%% dcnote difference between ccsm4RawData is no mu (time mean) subtraction here

function ccsm4DataRaw(xmin,xmax,ymin,ymax,yearmin,yearmax,lbl)

switch lbl
        
   case 'SST'
             
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.pop.h.TEMP.';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 10;                   % years per input file
%         idxX       = 20;                   % longitude index in nc file
%         idxY       = 19;                   % latitude index in nc file
%         idxT       = 57;                   % time index in nc file
%         idxA       = 17;                   % area index in nc file
%         idxM       = 15;                   % region mask
%         idxLbl     = 18;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );
        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
        nTBatch = yrBatch * 12;                         % number of months in each data batch
        
        while yrStart < yrLim( 2 )
            
            yrEnd = yrStart + yrBatch - 1;
            iTEnd = iTStart + yrBatch * 12 - 1;
            
            ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                sprintf('%04.0f', yrStart), '01-', ...
                sprintf('%04.0f', yrEnd),   '12.nc' ];
            
            disp( [ 'Reading input file ', ncFile ] )

            S_ocean_lon  = ncget( ncFile, 'TLONG');
            S_ocean_lat  = ncget( ncFile, 'TLAT');
            S_ocean_mask = ncget( ncFile, 'REGION_MASK');
            S_ocean_area = ncget( ncFile, 'TAREA');
            S_ocean_sst  = ncget( ncFile, 'TEMP'); % 3rd entry is z coordinate
            S_ocean_time = ncget( ncFile, 'time');   

            if iTStart == 1
                
                disp( 'Extracting gridpoints...' )

                ifOcean = S_ocean_mask ~= 0; % nonzero values are ocean gridpoints
              
                if(xLim(1)<xLim(2))

                    ifXY = S_ocean_lon >= xLim( 1 ) ...
                         & S_ocean_lon <= xLim( 2 ) ...
                         & S_ocean_lat >= yLim( 1 ) ...
                         & S_ocean_lat <= yLim( 2 ) ...
                         & ifOcean;

                end
              
                if(xLim(1)>xLim(2))

                    ifXY = (S_ocean_lon >= xLim( 1 ) ...
                          | S_ocean_lon <= xLim( 2 )) ...
                          & S_ocean_lat >= yLim( 1 ) ...
                          & S_ocean_lat <= yLim( 2 ) ...
                          & ifOcean;

                end
                
                nD       = nnz( ifXY );

                x        = S_ocean_lon;
                y        = S_ocean_lat;

                w        = double( S_ocean_area( ifXY ) );
                w        = sqrt( w / sum( w ) );
                
                disp( '' )           
                disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
                disp( [ num2str( max( w ) ),   ' max area weight' ] )
                disp( [ num2str( min( w ) ),   ' min area weight' ] )

                mu     = zeros( nD, 1 );
                myData = zeros( [ nD, nT ] );
                t      = zeros( 1, nT );

                gridFile    = [ dataDir, '/dataGrid.mat' ];
                save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

            end
            
            t( iTStart : iTEnd ) = S_ocean_time( 1 : end );
            
            for iT = 1 : nTBatch

                dataTmp = squeeze( S_ocean_sst( :, :, 1, iT ) );

                dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                 mu      = mu + dataTmp;
                myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
            
            end
            
            iTStart = iTEnd + 1;
            yrStart = yrEnd + 1;

        end

%         mu = mu / nT;
%         myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )


   case 'SIC'
        
       % dcmod - modified since all ice data in one file
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cice.h.aice_nh.000101-130012';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 1300;                   % years per input file
%         idxX       = 6;                   % longitude index in nc file
%         idxY       = 5;                   % latitude index in nc file
%         idxT       = 19;                   % time index in nc file
%         idxA       = 18;                   % area index in nc file
%         idxM       = 21;                   % region mask
%         idxLbl     = 9;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );

        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
%        nTBatch = yrBatch * 12;                         % number of months in each data batch

        yrEnd = yrLim(2);
        iTEnd = nT;

        ncFile = [ dataDirIn, '/', ...
            experiment, '/', ...
            fileBase, ...
            '.nc' ];
            
        disp( [ 'Reading input file ', ncFile ] )
        
        S_ice_lon  = ncget( ncFile, 'TLON');
        S_ice_lat  = ncget( ncFile, 'TLAT');
        S_ice_mask = ncget( ncFile, 'tmask');
        S_ice_area = ncget( ncFile, 'tarea');
        S_ice_aice  = ncget( ncFile, 'aice');
        S_ice_time = ncget( ncFile, 'time');     

        disp( 'Extracting gridpoints...' )

        ifIce = S_ice_mask ~= 0; % nonzero values are ocean gridpoints
              
        if(xLim(1)<xLim(2))
            
            ifXY = S_ice_lon >= xLim( 1 ) ...
                & S_ice_lon <= xLim( 2 ) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
              
        if(xLim(1)>xLim(2))
            
            ifXY = (S_ice_lon >= xLim( 1 ) ...
                | S_ice_lon <= xLim( 2 )) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end  

        nD       = nnz( ifXY ) 

        x        = S_ice_lon;
        y        = S_ice_lat;

        w        = double( S_ice_area( ifXY ) );
        w        = sqrt( w / sum( w ) );

        disp( '' )           
        disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
        disp( [ num2str( max( w ) ),   ' max area weight' ] )
        disp( [ num2str( min( w ) ),   ' min area weight' ] )

        mu     = zeros( nD, 1 );
        myData = zeros( [ nD, nT ] );
        t      = zeros( 1, nT );

        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

        %dcmod move to starting year
        t_adjust = (yrLim(1) - 1)*12
                
        t( iTStart : iTEnd ) = S_ice_time( iTStart + t_adjust : iTEnd + t_adjust);  

        for iT = 1 : nT
            
            dataTmp = squeeze( S_ice_aice( :, :, iT+t_adjust ) );          
            dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%             mu      = mu + dataTmp;
            myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
            
        end

%         mu = mu / nT;
%        myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
        
    case 'SIT'
        
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cice.h.hi_nh.000101-130012';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 1300;                   % years per input file
%         idxX       = 6;                   % longitude index in nc file
%         idxY       = 5;                   % latitude index in nc file
%         idxT       = 19;                   % time index in nc file
%         idxA       = 18;                   % area index in nc file
%         idxM       = 21;                   % region mask
%         idxLbl     = 9;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );

        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
%        nTBatch = yrBatch * 12;                         % number of months in each data batch

        yrEnd = yrLim(2);
        iTEnd = nT;

        ncFile = [ dataDirIn, '/', ...
            experiment, '/', ...
            fileBase, ...
            '.nc' ];
            
        disp( [ 'Reading input file ', ncFile ] )

        S_ice_lon  = ncget( ncFile, 'TLON');            
        S_ice_lat  = ncget( ncFile, 'TLAT');
        S_ice_mask = ncget( ncFile, 'tmask');
        S_ice_area = ncget( ncFile, 'tarea');
        S_ice_sit  = ncget( ncFile, 'hi'); % wrong order need to permute        
        S_ice_time = ncget( ncFile, 'time');        

        disp( 'Extracting gridpoints...' )

        ifIce = S_ice_mask ~= 0; % nonzero values are ocean gridpoints
              
        if(xLim(1)<xLim(2))
            
            ifXY = S_ice_lon >= xLim( 1 ) ...
                & S_ice_lon <= xLim( 2 ) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
              
        if(xLim(1)>xLim(2))
            
            ifXY = (S_ice_lon >= xLim( 1 ) ...
                | S_ice_lon <= xLim( 2 )) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
          

        nD       = nnz( ifXY ) 

        x        = S_ice_lon;
        y        = S_ice_lat;

        w        = double( S_ice_area( ifXY ) );
        w        = sqrt( w / sum( w ) );

        disp( '' )           
        disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
        disp( [ num2str( max( w ) ),   ' max area weight' ] )
        disp( [ num2str( min( w ) ),   ' min area weight' ] )

        mu     = zeros( nD, 1 );
        myData = zeros( [ nD, nT ] );
        t      = zeros( 1, nT );

        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

        %dcmod move to starting year
        t_adjust = (yrLim(1) - 1)*12
                
        t( iTStart : iTEnd ) = S_ice_time( iTStart + t_adjust : iTEnd + t_adjust);

        for iT = 1 : nT

            dataTmp = squeeze( S_ice_sit( :, :, iT+t_adjust ) );

            dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%             mu      = mu + dataTmp;
            myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
            
        end

%         mu = mu / nT;
%         myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
        

        
    case 'SLP_train'
        % only for years 100-499
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cam2.h0.PSL.';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 200;                   % years per input file
%         idxX       = 20;                   % longitude index in nc file
%         idxY       = 19;                   % latitude index in nc file
%         idxT       = 57;                   % time index in nc file
%         idxA       = 17;                   % area index in nc file
%         idxM       = 15;                   % region mask
%         idxLbl     = 18;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );
        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
        nTBatch = yrBatch * 12;                         % number of months in each data batch
        
        fileNum = 1; % hack to adjust here for start/end times not syncing with files
        
        while yrStart < yrLim( 2 )
            
%             yrEnd = yrStart + yrBatch - 1;
%             iTEnd = iTStart + yrBatch * 12 - 1;
            
%             ncFile = [ dataDirIn, '/', ...
%                 experiment, '/', ...
%                 fileBase, ...
%                 sprintf('%04.0f', yrStart), '01-', ...
%                 sprintf('%04.0f', yrEnd),   '12.nc' ];
            
            if fileNum == 1
                yrEnd = 100 + 100 - 1;
                iTEnd = iTStart + 100 * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '000101-019912.nc' ];
            
            elseif fileNum == 2
                yrEnd = yrStart + yrBatch - 1;
                iTEnd = iTStart + yrBatch * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '020001-039912.nc' ];
            
            elseif fileNum == 3
                yrEnd = yrStart + 100 - 1;
                iTEnd = iTStart + 100 * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '040001-059912.nc' ];
            
            end
            
            disp( [ 'Reading input file ', ncFile ] )

            S_atm_slp  = ncget( ncFile, 'PSL'); % lon x lat x time
            S_atm_time = ncget( ncFile, 'time');
            
            lon  = ncget( ncFile, 'lon'); % 288
            lat  = ncget( ncFile, 'lat'); % 192
            S_atm_lat = zeros(288,192);
            S_atm_lon = zeros(288,192);
            
            for i = 1:288
                for j = 1:192
                    S_atm_lon(i,j) = lon(i);
                    S_atm_lat(i,j) = lat(j);
                end
            end
            
            
                     
            if iTStart == 1
                
                disp( 'Extracting gridpoints...' )
              
                if(xLim(1)<xLim(2))

                    ifXY = S_atm_lon >= xLim( 1 ) ...
                         & S_atm_lon <= xLim( 2 ) ...
                         & S_atm_lat >= yLim( 1 ) ...
                         & S_atm_lat <= yLim( 2 );

                end
              
                if(xLim(1)>xLim(2))

                    ifXY = (S_atm_lon >= xLim( 1 ) ...
                          | S_atm_lon <= xLim( 2 )) ...
                          & S_atm_lat >= yLim( 1 ) ...
                          & S_atm_lat <= yLim( 2 );

                end
                
                nD       = nnz( ifXY );

                x        = S_atm_lon;
                y        = S_atm_lat;
                
                weights=cos(y*pi/180);
                w=weights( ifXY );
                w = sqrt( w / sum( w ) );
                
                disp( '' )           
                disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
                disp( [ num2str( max( w ) ),   ' max area weight' ] )
                disp( [ num2str( min( w ) ),   ' min area weight' ] )

                mu     = zeros( nD, 1 );
                myData = zeros( [ nD, nT ] );
                t      = zeros( 1, nT );

                gridFile    = [ dataDir, '/dataGrid.mat' ];
                save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

            end
            
            if fileNum == 1
                t_adjust = 99*12 + 1
                t( iTStart : iTEnd ) = S_atm_time( t_adjust : end );
                
                for iT = t_adjust : nTBatch - 12
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - t_adjust ) = dataTmp; %fill up columns of data matrix
                end
                
            elseif fileNum == 2
                t( iTStart : iTEnd ) = S_atm_time( 1 : end );
            
                for iT = 1 : nTBatch
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
                end
                
            elseif fileNum == 3
                t( iTStart : iTEnd ) = S_atm_time( 1 : end - 1200); 
                for iT = 1 : 1200
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - 1 ) = dataTmp; %fill up columns of data matrix
                end
                
            end 
            
            iTStart = iTEnd + 1;
            yrStart = yrEnd + 1;
            fileNum = fileNum + 1;

        end

%         mu = mu / nT;
%         myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
        
        
    case 'SLP_test'
        % only for years 500 - 899
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cam2.h0.PSL.';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 200;                   % years per input file
%         idxX       = 20;                   % longitude index in nc file
%         idxY       = 19;                   % latitude index in nc file
%         idxT       = 57;                   % time index in nc file
%         idxA       = 17;                   % area index in nc file
%         idxM       = 15;                   % region mask
%         idxLbl     = 18;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );
        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
        nTBatch = yrBatch * 12;                         % number of months in each data batch
        
        fileNum = 1; % hack to adjust here for start/end times not syncing with files
        
        while yrStart < yrLim( 2 )
            
%             yrEnd = yrStart + yrBatch - 1;
%             iTEnd = iTStart + yrBatch * 12 - 1;
            
%             ncFile = [ dataDirIn, '/', ...
%                 experiment, '/', ...
%                 fileBase, ...
%                 sprintf('%04.0f', yrStart), '01-', ...
%                 sprintf('%04.0f', yrEnd),   '12.nc' ];
            
            if fileNum == 1
                yrEnd = 500 + 100 - 1;
                iTEnd = iTStart + 100 * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '040001-059912.nc' ];
            
            elseif fileNum == 2
                yrEnd = yrStart + yrBatch - 1;
                iTEnd = iTStart + yrBatch * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '060001-079912.nc' ];
            
            elseif fileNum == 3
                yrEnd = yrStart + 100 - 1;
                iTEnd = iTStart + 100 * 12 - 1;
                ncFile = [ dataDirIn, '/', ...
                experiment, '/', ...
                fileBase, ...
                '080001-099912.nc' ];
            
            end
            
            disp( [ 'Reading input file ', ncFile ] )

            S_atm_slp  = ncget( ncFile, 'PSL'); % lon x lat x time
            S_atm_time = ncget( ncFile, 'time');
            
            lon  = ncget( ncFile, 'lon'); % 288
            lat  = ncget( ncFile, 'lat'); % 192
            S_atm_lat = zeros(288,192);
            S_atm_lon = zeros(288,192);
            
            for i = 1:288
                for j = 1:192
                    S_atm_lon(i,j) = lon(i);
                    S_atm_lat(i,j) = lat(j);
                end
            end
            
            
                     
            if iTStart == 1
                
                disp( 'Extracting gridpoints...' )
              
                if(xLim(1)<xLim(2))

                    ifXY = S_atm_lon >= xLim( 1 ) ...
                         & S_atm_lon <= xLim( 2 ) ...
                         & S_atm_lat >= yLim( 1 ) ...
                         & S_atm_lat <= yLim( 2 );

                end
              
                if(xLim(1)>xLim(2))

                    ifXY = (S_atm_lon >= xLim( 1 ) ...
                          | S_atm_lon <= xLim( 2 )) ...
                          & S_atm_lat >= yLim( 1 ) ...
                          & S_atm_lat <= yLim( 2 );

                end
                
                nD       = nnz( ifXY );

                x        = S_atm_lon;
                y        = S_atm_lat;
                
                weights=cos(y*pi/180);
                w=weights( ifXY );
                w = sqrt( w / sum( w ) );
                
                disp( '' )           
                disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
                disp( [ num2str( max( w ) ),   ' max area weight' ] )
                disp( [ num2str( min( w ) ),   ' min area weight' ] )

                mu     = zeros( nD, 1 );
                myData = zeros( [ nD, nT ] );
                t      = zeros( 1, nT );

                gridFile    = [ dataDir, '/dataGrid.mat' ];
                save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

            end
            
            if fileNum == 1
                t_adjust = 100*12 + 1
                t( iTStart : iTEnd ) = S_atm_time( t_adjust : end );
                
                for iT = t_adjust : nTBatch
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - t_adjust ) = dataTmp; %fill up columns of data matrix
                end
                
            elseif fileNum == 2
                t( iTStart : iTEnd ) = S_atm_time( 1 : end );
            
                for iT = 1 : nTBatch
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
                end
                
            elseif fileNum == 3
                t( iTStart : iTEnd ) = S_atm_time( 1 : end - 1200); 
                for iT = 1 : 1200
                    dataTmp = squeeze( S_atm_slp( :, :, iT ) );
                    dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%                     mu      = mu + dataTmp;
                    myData( :, iTStart + iT - 1 ) = dataTmp; %fill up columns of data matrix
                end
                
            end 
            
            iTStart = iTEnd + 1;
            yrStart = yrEnd + 1
            fileNum = fileNum + 1;

        end

%         mu = mu / nT;
%         myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )     
        
        
        
   case 'SIA'
        
       % dcmod - modified since all ice data in one file
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cice.h.aice_nh.000101-130012';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 1300;                   % years per input file
%         idxX       = 6;                   % longitude index in nc file
%         idxY       = 5;                   % latitude index in nc file
%         idxT       = 19;                   % time index in nc file
%         idxA       = 18;                   % area index in nc file
%         idxM       = 21;                   % region mask
%         idxLbl     = 9;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );

        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
%        nTBatch = yrBatch * 12;                         % number of months in each data batch

        yrEnd = yrLim(2);
        iTEnd = nT;

        ncFile = [ dataDirIn, '/', ...
            experiment, '/', ...
            fileBase, ...
            '.nc' ];
            
        disp( [ 'Reading input file ', ncFile ] )
        
        S_ice_lon  = ncget( ncFile, 'TLON');
        S_ice_lat  = ncget( ncFile, 'TLAT');
        S_ice_mask = ncget( ncFile, 'tmask');
        S_ice_area = ncget( ncFile, 'tarea');
        S_ice_aice  = ncget( ncFile, 'aice');
        S_ice_time = ncget( ncFile, 'time');     

        disp( 'Extracting gridpoints...' )

        ifIce = S_ice_mask ~= 0; % nonzero values are ocean gridpoints
              
        if(xLim(1)<xLim(2))
            
            ifXY = S_ice_lon >= xLim( 1 ) ...
                & S_ice_lon <= xLim( 2 ) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
              
        if(xLim(1)>xLim(2))
            
            ifXY = (S_ice_lon >= xLim( 1 ) ...
                | S_ice_lon <= xLim( 2 )) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end  

        nD       = nnz( ifXY ) 

        x        = S_ice_lon;
        y        = S_ice_lat;

        w        = double( S_ice_area( ifXY ) );
%         w        = sqrt( w / sum( w ) );

        disp( '' )           
        disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
        disp( [ num2str( max( w ) ),   ' max area weight' ] )
        disp( [ num2str( min( w ) ),   ' min area weight' ] )

        mu     = zeros( nD, 1 );
        myData = zeros( [ nD, nT ] );
        t      = zeros( 1, nT );

        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

        %dcmod move to starting year
        t_adjust = (yrLim(1) - 1)*12
                
        t( iTStart : iTEnd ) = S_ice_time( iTStart + t_adjust : iTEnd + t_adjust);  

        for iT = 1 : nT
            
            dataTmp = squeeze( S_ice_aice( :, :, iT+t_adjust ) );          
            dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%             mu      = mu + dataTmp;
            myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
            
        end

%         mu = mu / nT;
%        myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
        
    case 'SIT_UW'
        
        dataSet='CCSM4_Data';
        dataDirIn  = '/kontiki_array5/data/ccsm4';
        experiment = 'b40.1850'; 
        fileBase   = 'b40.1850.track1.1deg.006.cice.h.hi_nh.000101-130012';
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years) 
        yrBatch    = 1300;                   % years per input file
%         idxX       = 6;                   % longitude index in nc file
%         idxY       = 5;                   % latitude index in nc file
%         idxT       = 19;                   % time index in nc file
%         idxA       = 18;                   % area index in nc file
%         idxM       = 21;                   % region mask
%         idxLbl     = 9;                   % field index in nc file
        
        dataDir = [ './data/raw/', ...
            dataSet,'/'...
            experiment, '/'...
            lbl, '/', ...
            'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
            '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
            '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
            '/data' ];
        
        mkdir( dataDir )
        
        yrStart = yrLim( 1 );

        iTStart = 1;
        
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
%        nTBatch = yrBatch * 12;                         % number of months in each data batch

        yrEnd = yrLim(2);
        iTEnd = nT;

        ncFile = [ dataDirIn, '/', ...
            experiment, '/', ...
            fileBase, ...
            '.nc' ];
            
        disp( [ 'Reading input file ', ncFile ] )

        S_ice_lon  = ncget( ncFile, 'TLON');            
        S_ice_lat  = ncget( ncFile, 'TLAT');
        S_ice_mask = ncget( ncFile, 'tmask');
        S_ice_area = ncget( ncFile, 'tarea');
        S_ice_sit  = ncget( ncFile, 'hi'); % wrong order need to permute        
        S_ice_time = ncget( ncFile, 'time');        

        disp( 'Extracting gridpoints...' )

        ifIce = S_ice_mask ~= 0; % nonzero values are ocean gridpoints
              
        if(xLim(1)<xLim(2))
            
            ifXY = S_ice_lon >= xLim( 1 ) ...
                & S_ice_lon <= xLim( 2 ) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
              
        if(xLim(1)>xLim(2))
            
            ifXY = (S_ice_lon >= xLim( 1 ) ...
                | S_ice_lon <= xLim( 2 )) ...
                & S_ice_lat >= yLim( 1 ) ...
                & S_ice_lat <= yLim( 2 ) ...
                & ifIce;
        
        end
          

        nD       = nnz( ifXY ) 

        x        = S_ice_lon;
        y        = S_ice_lat;

        w        = double( S_ice_area( ifXY ) );
        w        = sqrt( w / sum( w ) );

        disp( '' )           
        disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
        disp( [ num2str( max( w ) ),   ' max area weight' ] )
        disp( [ num2str( min( w ) ),   ' min area weight' ] )

        mu     = zeros( nD, 1 );
        myData = zeros( [ nD, nT ] );
        t      = zeros( 1, nT );

        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD' )  

        %dcmod move to starting year
        t_adjust = (yrLim(1) - 1)*12
                
        t( iTStart : iTEnd ) = S_ice_time( iTStart + t_adjust : iTEnd + t_adjust);

        for iT = 1 : nT

            dataTmp = squeeze( S_ice_sit( :, :, iT+t_adjust ) );

            dataTmp = double( dataTmp( ifXY ) ); %data is unweighted
%             dataTmp = double( dataTmp( ifXY ) ).*w; %data is area weighted
%             mu      = mu + dataTmp;
            myData( :, iTStart + iT - 1  ) = dataTmp; %fill up columns of data matrix
            
        end

%         mu = mu / nT;
%         myData = bsxfun( @minus, myData, mu );%subtract time mean from data

        l2NormT = sum( myData .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );
        
        x = zeros( [ nD, nT ] );
        x=myData;

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
        
           
end



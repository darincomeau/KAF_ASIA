function nsidcRawData(xmin,xmax,ymin,ymax,yearmin,yearmax,lbl)

        addpath('./data/processed_data/m_map')

        dataSet='nsidc';
        dataDirIn ='../../../../kontiki_array5/data';
        
        xLim       = [ xmin xmax  ];       % longitude limits
        yLim       = [ ymin  ymax ];       % latitude limits
        yrLim      = [ yearmin yearmax ];  % time limits (years)
        
        dataDir = [ './data/raw/', ...
             dataSet,'/'...
		     lbl,'/', ...
		     'x',   int2str( xLim( 1 ) ),  '-', int2str( xLim( 2 ) ), ...
		     '_y',  int2str( yLim( 1 ) ),  '-', int2str( yLim( 2 ) ), ...
		     '_yr', int2str( yrLim( 1 ) ), '-', int2str( yrLim( 2 ) ), ...
		     '/dataAnomaly' ];
         
        mkdir( dataDir )
       
        nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months

        
%Get grid first
     
        fileBase='pub/DATASETS/seaice/polar-stereo/tools';
        
        %get lon
        
        fid=fopen([dataDirIn,'/',dataSet,'/',fileBase,'/','psn25lons_v3.dat']);%divide by 100000 to get lon
        
        a=fread(fid,'int32');
       
        b=reshape(a,304,448);

        lon=b/100000;
      
        figure(1)
        pcolor(lon)
        shading interp
        colorbar
                                    
        %get lat
        
        fid=fopen([dataDirIn,'/',dataSet,'/',fileBase,'/','psn25lats_v3.dat']);%divide by 100000 to get lon
        
        a=fread(fid,'int32');
        
        b=reshape(a,304,448);

        lat=b/100000;
        
        figure(2)
        pcolor(lat)
        shading interp
        colorbar 
                
        %get area
        
        fid=fopen([dataDirIn,'/',dataSet,'/',fileBase,'/','psn25area_v3.dat']);%values are quoted as max of 6.5e5. Multiply by 1000 to get m^2.
        
        a=fread(fid,'int32');
        
        b=reshape(a,304,448);

        area=b*1000;
        
        figure(3)
        pcolor(lon,lat,area)
        shading interp
        colorbar
        
        %%Get data
        
        dataSet='nsidc';
        dataDirIn ='../../../../kontiki_array5/data';
        
        %Get grid first
     
        fileBase='north/pub/DATASETS/nsidc0079_gsfc_bootstrap_seaice/final-gsfc/north/monthly';
        
        counter=1;
        
        data=zeros(size(lon,1),size(lon,2),(2013-1979+1)*12);
        
        for year=1979:1:2013
            for month=1:1:12
                
                if(counter<12*(1987-1979)+8)
                    sat='n07';
                end
                
                if(counter>=12*(1987-1979)+8 && counter <12*(1991-1979)+12) 
                    sat='f08';
                end
                
                if(counter>=12*(1991-1979)+12 && counter <(1995-1979)*12+10)
                    sat='f11';
                end
                
                if(counter>=(1995-1979)*12+10 && counter <(2007-1979)*12+13)
                    sat='f13';
                end
                
                if(counter >=(2007-1979)*12+13)
                    sat='f17';
                end
                
                fname=[dataDirIn,'/',dataSet,'/',fileBase,'/','bt_',num2str(year),...
                    sprintf('%02d',month),'_',sat,'_v02_n.bin']
        
            fid=fopen(fname);
        
            a=fread(fid,'int16');
        
            b=reshape(a,304,448);

            sic=b/10;
            
            data(:,:,counter)=sic;
            
            counter=counter+1;
            
            end
        end
        
        nT=size(data,3);
        
%         figure(4)
%         
%         for i=1:nT
%     
%                m_proj('stereographic','lat',90,'long',30,'radius',45);
%                h = m_pcolor( lon, lat, data(:,:,i) );
%                shading interp
%                %caxis([0 50])
%                colorbar
%                m_grid('xtick',12,'tickdir','out','ytick',[45:15:90],'linest','-');
%                m_coast( 'line', 'linewidth', 1, 'color', 'k' );
%                drawnow
%                
%                set(gcf,'color', 'w')
%         end

x=lon;
y=lat;

ifOcean=(data(:,:,1)<1001);
x_new=mod(x,360);

              if(xLim(1)<xLim(2))

                  ifXY = x_new >= xLim( 1 ) ...
                  & x_new <= xLim( 2 ) ...
                  & y >= yLim( 1 ) ...
                  & y <= yLim( 2 ) ...
                  & ifOcean;    

              end
              
              if(xLim(1)>xLim(2))

                  ifXY = (x_new >= xLim( 1 ) ...
                  | x_new <= xLim( 2 )) ...
                  & y >= yLim( 1 ) ...
                  & y <= yLim( 2 ) ...
                  & ifOcean;  
                                    
              end
                 

        nD       = nnz( ifXY ); 

        w        = double( area( ifXY ) );
        w        = sqrt( w / sum( w ) );

        %disp( [ 'Latitude label  ',      S_ice.VarArray( idxY ).Str ] )
        %disp( [ 'Longitude label ',      S_ice.VarArray( idxX ).Str ] )
        %disp( [ 'Area label      ',      S_ice.VarArray( idxA ).Str ] )
        %disp( [ 'Mask label      ',      S_ice.VarArray( idxM ).Str ] )
        %disp( [ 'Field label     ',      S_ice.VarArray( idxLbl ).Str ] )
        disp( '' )           
        disp( [ int2str( nD ),         ' unmasked gridpoints (physical space dimension)' ] )
        disp( [ num2str( max( w ) ),   ' max area weight' ] )
        disp( [ num2str( min( w ) ),   ' min area weight' ] )
        
        %%%edit below this point
        
        mu     = zeros( nD, 1 );
        myData = zeros( [ nD, nT ] );
        t      = zeros( 1, nT );

        gridFile = [ dataDir, '/dataGrid.mat'];
        
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'lbl', 'nD')

        time = yrLim(1)+1/12:1/12:yrLim(2);
        
        for iT = 1 : nT

                dataTmp = data( :, :, (yrLim(1)-1)*12+iT);

                dataTmp = double( dataTmp( ifXY ) );
                mu      = mu + dataTmp;
                myData( :, iT ) = dataTmp;
        end

        mu = mu / nT;
        myData = bsxfun( @minus, myData, mu );

        for i=1:nT
            myData(:,i)=myData(:,i).*w;%area weighting 
        end

        x = zeros( [ nD, nT ] );
        x=myData;
       
        time=1:nT;
        
        lin_trend=zeros(nD,nT);
               
           for i=1:nD
                
                a=x(i,:);

               
                p=polyfit(time,a,1);
                
                L=p(1)*time+p(2);
                  
                lin_trend(i,:)=L;
                i
                
           end
            
         seasonal_trend=zeros(nD,nT);
         
         for month=1:12
             
             month
             
            for i=1:nD
               
                a=x(i,month:12:end);
                
                time=1:length(a);
                
                p=polyfit(time,a,1);
                
                L=p(1)*time+p(2);
                  
                seasonal_trend(i,month:12:end)=L-mean(L);
                
            end
            
         end
 
        v=sum(var(x,0,2))/nD

        
        l2NormT = sum( x .^ 2, 1 );
        l2Norm  = sqrt( sum( l2NormT ) / nT );
        l2NormT = sqrt( l2NormT );

        anomFile    = [ dataDir, '/dataX.mat' ];
        save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm','v','lin_trend', 'seasonal_trend')
 

        
        
        



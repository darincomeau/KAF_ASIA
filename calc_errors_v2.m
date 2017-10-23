function [ pred_rms, pred_pc, pred_rmsP, pred_pcP ] =  calc_errors_v2( pred_traj, truth )

% produces rms and pc error metrics for forecasts and persistence (P)

[nIter, tLag] = size(pred_traj);

pred_rms   = zeros(1,tLag);
pred_pc    = zeros(1,tLag);
mean_pred  = zeros(1,tLag);
std_pred   = zeros(1,tLag);
mean_truth = zeros(1,tLag);
std_truth  = zeros(1,tLag);

predP      = zeros(size(pred_traj));
pred_rmsP  = zeros(1,tLag);
pred_pcP   = zeros(1,tLag);
meanP      = zeros(1,tLag);
stdP       = zeros(1,tLag);

%% forecast errors
% rms
for i = 1:tLag
    for j = 1:nIter
        pred_rms(i) = pred_rms(i) + (pred_traj(j,i) - truth(j,i))^2;
    end
    pred_rms(i) = sqrt(pred_rms(i)/nIter);
end

% pattern correlation

for i = 1:tLag
    mean_pred(i) = mean(pred_traj(:,i));
    std_pred(i) = std(pred_traj(:,i));
       
    mean_truth(i) = mean(truth(:,i));
    std_truth(i)  =  std(truth(:,i));
    
    for j = 1:nIter
        pred_pc(i) = pred_pc(i) + ...
                    (pred_traj(j,i)-mean_pred(i))*(truth(j,i)-mean_truth(i));
    end
    pred_pc(i) = pred_pc(i)/(std_pred(i)*std_truth(i)*nIter);
end


%% persistence errors
% note persistence is based on truth
for j = 1:nIter
        predP(j,:) = truth(j,1);
end

% rms
for i = 1:tLag
    for j = 1:nIter
        pred_rmsP(i) = pred_rmsP(i) + (predP(j,i) - truth(j,i))^2;
    end
    pred_rmsP(i) = sqrt(pred_rmsP(i)/nIter);
end

% pattern correlation
for i = 1:tLag
    meanP(i) = mean(predP(:,i));
    stdP(i) = std(predP(:,i));
    
    for j = 1:nIter
        pred_pcP(i) = pred_pcP(i) + ...
                    (predP(j,i)-meanP(i))*(truth(j,i)-mean_truth(i));
    end
    pred_pcP(i) = pred_pcP(i)/(stdP(i)*std_truth(i)*nIter);
end

end

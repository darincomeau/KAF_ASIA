function [ d ] = LP_precomp( d_ref, f_ref, tau, eps_0, max_iter )
%This function computes the multiscale approximation on the test data set
%Input:
%   d_ref is a distance matrix of size nxn
%   f_ref is a matrix of size (n+tau-1)x1, Note the length of f_ref. tau = 1 means current time.
%   tau: forecast time
%   eps_0: initial kernel scale
%   max_iter: maximum number of iterations. default set to be 15
%Output:
%   d: decomposed function
%   This function uses distance defined by users. It can be Euclidean
%   distance, diffusion map distance or other metrics.
%   Zhizhen Zhao 03/31/2016

%   d is d{tau}(nIter,level)

if nargin == 4
   max_iter = 15;
end

n = size(d_ref, 1); %number of training data points
id_ref = find(d_ref == 0); %if d_ref is a sparse matrix

%Decompose the function on reference set into multiple scales
counter = 0;
d = cell(tau, 1);
%intial data with all zeros
 for i = 1:tau
     d{i} = zeros(n, max_iter);
 end;
f = zeros(n, tau);
for i = 1:max_iter
    eps = eps_0/(2^(i-1));
    tmp = exp(-d_ref/eps);
    tmp(id_ref) = 0;
    D = sum(tmp, 2);
    K = bsxfun(@times, tmp, 1./D);%    
    for k = 1:tau
%         tmp = f_ref(tau:n+tau-1) - f(:, k);
        tmp = f_ref(k:n+k-1) - f(:, k);        
        d{k}(:,i) = tmp;
        f(:, k) = f(:, k) + K*tmp;
        err(k, i) = norm(tmp, 'fro');
    end;
end

%Find optimal stopping for each lead time
 for k = 1:tau
     tmp = err(k, 2:end)-err(k, 1:end-1);
     index = find(tmp>=0, 1);
     d{k} = d{k}(:, 1:index);
 end;



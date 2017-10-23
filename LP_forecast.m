function [ f_ext ] = LP_forecast( d_ose, eps_0, d )

%This function computes function extension on the test data set
%Input:
%   d_ref is a distance matrix of size nxn
%   eps_0: initial kernel scale
%   d: decomposed function from LP_precomp.m
%Output:
%   f_ext: LP forecast
%   This function uses distance defined by users. It can be Euclidean
%   distance, diffusion map distance or other metrics.
%   Zhizhen Zhao 03/31/2016

[ m, n ] = size(d_ose);
id_ose = find(d_ose == 0);
tau = size( d, 1 );
f_ext = zeros(m, tau);
num_iter = zeros(tau, 1);

 for k = 1:tau
     num_iter(k) = size(d{k}, 2);
 end;
 max_iter = max(num_iter)


for i = 1:max_iter
    eps = eps_0/(2^(i-1));
    tmp = exp(-d_ose/eps);
    tmp(id_ose) = 0;
    D = sum(tmp, 2);
    K = bsxfun(@times, tmp, 1./D);
    for k = 1:tau
        if i<=num_iter(k)
             f_ext(:, k) = f_ext(:, k) + K*d{k}(:, i);
        end;
    end;
end;   



function [ dist ] = mmd( sourcefeatures,targetfeatures,sigma )
%MMD Summary of this function goes here
%   This code computes the empirical estimate of the distance between
%   distributions P (source) and Q (target), as defined by Maximum Mean
%   Discrepancy.
%   
%   Dist(Xs,Xt) = sqrt(tr(KL)
%   K = [Kss Kst
%        Kts Ktt]
%   L = 1/n1^2(i,j belong to Xs), = 1/n2^2(i,j belong to Xt), = -1/(n1n2)
%   (otherwise)
    
    %% compute K
    Kss = rbf_dot(sourcefeatures,sourcefeatures,sigma);
    Kst = rbf_dot(sourcefeatures,targetfeatures,sigma);
    Kts = rbf_dot(targetfeatures,sourcefeatures,sigma);
    Ktt = rbf_dot(targetfeatures,targetfeatures,sigma);
    K = [[Kss,Kst];[Kts,Ktt]];
    
    %% compute L
    n1 = size(sourcefeatures,1);
    n2 = size(targetfeatures,1);
    L = zeros(n1+n2);
    L(1:n1,1:n1) = 1/(n1^2);
    L(n1+1:end,n1+1:end) = 1/(n2^2);
    L(n1+1:end,1:n1) = -1/(n1*n2);
    L(1:n1,n1+1:end) = -1/(n1*n2);
    
    %% compute dist
    dist = sqrt(trace(K*L));
    
end


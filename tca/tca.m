function [Kr] = tca( sourcefeatures,targetfeatures,m,sigma)
%TCA Summary of this function goes here
%   This is the code for 'Domain Adaptation via Transfer Component Analysis' by Sinno Jialin Pan, Ivor W. Tsang, James T. Kwok and Qiang Yang
%   input: source features, target features, the number of vectors after
%   tca, and the parameter sigma for gaussian kernel.
%   output: resultant kernel matrix
    %% error
    if m >= size(sourcefeatures,2)
        error 'm is larger than the total number of features';
    end
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
    
    %% compute u and H
    u = 0.1; %trade-off parameter
    H = eye(n1+n2) - ones(n1+n2)/(n1+n2); %centering matrix
    
    %% calculate m-leading eigenvalues and corresponding eigenvectors
    I = eye(n1+n2);
    M = (I + u*K*L*K)\K*H*K;
    [V,D] = eig(M);
    Dreal = real(diag(D));
    [Dreal,indice] = sort(Dreal,'descend');
    W = V(:,indice(1:m));
    
    %% calculate resultant kernel matrix
    Kr = K*W*W'*K;
end


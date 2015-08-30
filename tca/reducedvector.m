function [features] = reducedvector(K,m)
%REDUCEDVECTOR Summary of this function goes here
%   This function returns the data of reduced feature space

    %% getting the m-leading eigenvectors and eigenvalues of K
    [V,D] = eig(K);
    Dreal = real(diag(D));
    [Dreal,indice] = sort(Dreal,'descend');
    V = V(:,indice(1:m));
    
    %% compute the phi(xi) using eigenvector and eigenvalues
    features = zeros(size(Dreal),m);
    for i=1:size(Dreal)
        for j=1:m
            features(i,j) = sqrt(Dreal(j)) * V(i,j);
        end
    end
end


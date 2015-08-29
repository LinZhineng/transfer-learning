%Radial basis function inner product
%Arthur Gretton

%Pattern input format : [pattern1 ; pattern2 ; ...]
%Output : Matrix of RBF values k(x1,x2)
%Deg is kernel size


function [H]=rbf_dot(patterns1,patterns2,deg)

%Note : patterns are transposed for compatibility with C code.

size1=size(patterns1);
size2=size(patterns2);


G = sum((patterns1.*patterns1),2);
H = sum((patterns2.*patterns2),2);

Q = repmat(G,1,size2(1));
R = repmat(H',size1(1),1);

H = Q + R - 2*patterns1*patterns2';


H=exp(-H/2/deg^2);
% 
% function K = rbf_dot(X, Y,rbf_var)
% 
% % Rows of X and Y are data points
% 
% xnum = size(X,1);
% 
% ynum = size(Y,1);
% 
% %if (kernel == 1) % Apply Gaussian kernel
%     for i=1:xnum
%    %     fprintf('i=%d\n',i);
%         for j=1:ynum
%             K(i,j) = exp(-norm(X(i,:)-Y(j,:))^2/rbf_var);
%            % K(i,j) = X(i,:)*Y(j,:)'; 
%         end
%     end
% 
% % % elseif(kernel==2) % Apply linear kernel
%   %   K = X*Y';
% % elseif(kernel==2) %polynomial kernel
% %     K =

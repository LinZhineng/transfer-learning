clear all;
addpath('./toolbox/libsvm-3.20/matlab');
load ../code/features;
K = tca(source.trn.features,target.test.features,20,10);
n = length(source.trn.labels);
m = length(target.test.labels);
%% if directly use the resultant kernel to do the training
model = svmtrain(source.trn.labels,[(1:n)',K(1:n,1:n)],'-t 4');
label = svmpredict(target.test.labels,[(1:m)',K(n+1:end,1:n)],model);

%% or you can use the data with reduced feature space to do the training
features = reducedvector(K,20);
sourcefeatures = features(1:n,:);
targetfeatures = features(n+1:end,:);
model = svmtrain(source.trn.labels,sourcefeatures,'-t 2');
label = svmpredict(target.test.labels,targetfeatures,model);
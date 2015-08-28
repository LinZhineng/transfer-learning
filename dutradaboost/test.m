clear all;
addpath('./toolbox/libsvm-weights-3.20/matlab');
load ../code/features.mat;
[model,beta] = DuTrAdaBoostTrain(source.trn.features,source.trn.labels,target.trn.features([1:3,target.trn.negImgs+1:target.trn.negImgs+3],:),target.trn.labels([1:3,target.trn.negImgs+1:target.trn.negImgs+3],:));
label = DuTrPredict(target.test.features, model, beta);
rightcount0 = 0;
rightcount1 = 0;
total0 = 0;
total1 = 0;
label0 = 0;
label1 = 0;
for i=1:length(target.test.labels)
    if label(i) == target.test.labels(i)
        if label(i)==-1
            rightcount0 = rightcount0 + 1;
        else
            rightcount1 = rightcount1 + 1;
        end
    end
    if target.test.labels(i) == -1
        total0 = total0 + 1;
    else
        total1 = total1 + 1;
    end
    if label(i) == -1
        label0 = label0 + 1;
    else
        label1 = label1 + 1;
    end
end
rightrate0 = rightcount0/total0
rightrate1 = rightcount1/total1
labelrate0 = rightcount0/label0
labelrate1 = rightcount1/label1
rightrate = (rightcount0 + rightcount1)/(total0 + total1)
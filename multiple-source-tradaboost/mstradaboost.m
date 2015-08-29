function  [hyp,alpha] = mstradaboost(source,targettrnfeatures,targettrnlabels)
    %% source.trn.labels : labels of source domain 
    %% source.trn.features : features of source domain
    
    M = 10;  %%iteration number
    N = length(source);
    ns = 0;
    for i = 1:N
        ns = ns + length(source(i).trn.labels);
        ws(i).weight = ones(length(source(i).trn.labels),1);
    end
    m = length(targettrnlabels);
    as = log(1 + sqrt(2* log(ns/M)))/2;
    wt.weight = ones(length(targettrnlabels),1);
    sW = sum(wt.weight);
    hyp = {};
    er = ones(M,1);
    
    %% begin of iteration
    for t = 1:M
        model = {};
        et = ones(N,1);
        bestaccuracy = 0;
        for k = 1:N
            W = [ws(k).weight;wt.weight];
            X = [source(k).trn.features;targettrnfeatures];
            Y = [source(k).trn.labels;targettrnlabels];
%             for l =1:length(Y)
%                 if Y(l) == -1
%                     W(l) = W(l)*1.3;
%                 end
%             end
            model{k} = svmtrain(W,Y,X,'-t 2');
            [predict,accuracy,prob] = svmpredict(Y,X,model{k});
            n = length(source(k).trn.labels);
            et(k) = sum(wt.weight.*(predict(n+1:m+n)~=targettrnlabels)/sW);
            %% choose the model with least error rate and best accuracy
            if et(k) < er(t)
               er(t) = et(k);
               bestaccuracy = accuracy(1);
               hyp{t} = model{k};
            elseif et(k)==er(t)
                    if accuracy(1) > bestaccuracy
                        bestaccuracy = accuracy(1);
                        hyp{t} = model{k};
                    end
            end
        end
%         F = [];
%         js = 0;
%         for k=1:N
%             if et(k) == er(t)
%                 js = js+1;
%                 F(js) = k;
%             end
%         end
%         k = ceil(rand()*js);
%         if k==0
%             k = 1;
%         end
%         hyp{t} = model{F(k)};
        if er(t) > 0.5
            er(t) = 0.499;
        end
        if er(t) == 0
            er(t) = 0.001;
        end
        alpha(t) = log((1-er(t))/er(t))/2;
        
        %% updating weights
        for k=1:N
            predict = svmpredict(source(k).trn.labels,source(k).trn.features,hyp{t});
            n = length(ws(k).weight);
            for j=1:n
                ws(k).weight(j) = ws(k).weight(j)*exp(-as*abs(predict(j)-source(k).trn.labels(j)));
            end
        end
        predict = svmpredict(targettrnlabels,targettrnfeatures,hyp{t});
        for j=1:m
            wt.weight(j) = wt.weight(j)*exp(alpha(t)*abs(predict(j)-targettrnlabels(j)));
        end
    end
end

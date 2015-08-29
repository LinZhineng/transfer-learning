function [model, beta ] = DuTrAdaboostTrain(tdX,tdY,tsX,tsY)

    %% tdX: features of source domain
    %% tdY: labels of source domain
    %% tsX: features of target domain
    %% tsY: labels of target domain
    
    tX = [tdX ; tsX];
    tY = [tdY ; tsY];
    n = size(tdY,1);
    m = size(tsY,1);
    T = 20;  %iteration number
    w = ones(m+n,1);
    model = cell(1,T);
    beta = zeros(1,T);
    bsrc = 1/(1+sqrt(2*log(n)/T));
    for t = 1:T
        %p = w./(sum(abs(w)));
        model{t} = svmtrain(w,tY,tX,'-t 0'); % using linear kernel
        predict = svmpredict(tY,tX,model{t});
        sW = sum(w(n+1:m+n));
        et = sum(w(n+1:m+n).*(predict(n+1:m+n)~=tsY))/sW;
        if et >= 0.5
            et = 0.499;
        elseif et == 0
            et = 0.001;
        end        
        beta(t) = et/(1-et);
        Ct = 2*(1-et);
        wUpdate = [(Ct*bsrc*ones(n,1)).^(predict(1:n)~=tdY) ; (beta(t)*ones(m,1)).^(-(predict(n+1:m+n)~=tsY)) ];
        w = w.*wUpdate;
    end
end


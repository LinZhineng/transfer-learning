function [model, beta ] = TrAdaboostTrain(tdX,tdY,tsX,tsY)
    %%tdX: features from source domain
    %%tdY: labels from source domain
    %%tsX: features from target domain
    %%tsY: labels from target domain

    tX = [tdX ; tsX];
    tY = [tdY ; tsY];
    n = size(tdY,1);
    m = size(tsY,1);
    T = 20;
    w = ones(m+n,1);
    model = cell(1,T);
    beta = zeros(1,T);
    for t = 1:T
        p = w./(sum(abs(w)));
        model{t} = svmtrain(p,tY,tX,'-t 0');
        predict = svmpredict(tY,tX,model{t});
        sW = sum(w(n+1:m+n));
        et = sum(w(n+1:m+n).*(predict(n+1:m+n)~=tsY)/sW);
        if (et >= 0.5)
            et = 0.5;
        end
        bT = et/(1-et);
        beta(t) =bT;
        b = 1/(1+sqrt(2*log(n/T)));
        wUpdate = [(b*ones(n,1)).^(predict(1:n)~=tdY) ; (bT*ones(m,1)).^(-(predict(n+1:m+n)~=tsY)) ];
        w = w.*wUpdate;
    end
end


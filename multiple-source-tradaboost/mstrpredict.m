function label = mstrpredict(X, hyp, alpha)
    M = length(hyp);
    l = size(X,1);
    label = zeros(l,1);
    for i=1:M
        label = label + (alpha(i)*svmpredict(label,X,hyp{i}));
    end
    label = sign(label);

%     M = length(hyp);
%     start = ceil(M/2);
%     l = size(X,1);
%     label = zeros(l,1);
%     for i=1:M
%         label = label + (alpha(i)*svmpredict(label,X,hyp{i}));
%     end
%     maxlabel = 0;
%     for i=1:length(label)
%         if abs(label(i))>maxlabel
%             maxlabel = abs(label(i));
%         end
%     end
%     trainIndex = [];
%     testIndex = [];
%     for i=1:length(label)
%         if abs(label(i)) == maxlabel
%             label(i) = sign(label(i));
%             trainIndex = [trainIndex,i];
%         else
%             label(i) = 0;
%             testIndex = [testIndex,i];
%         end
%     end
%     w = ones(length(trainIndex),1);
%     model = svmtrain(w,label(trainIndex),X(trainIndex,:),'-t 0');
%     label2 = svmpredict(zeros(length(testIndex),1),X(testIndex,:),model);
%     label(testIndex) = sign(label2);
    
end


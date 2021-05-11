function [trainACC,testACC] = SVM(dataset,PredictorNames,ResponseName,args)
arguments
    dataset(1,:) char;
    PredictorNames(1,:) cell;
    ResponseName(1,:) char;
    args(1,:) cell;
end

load(dataset);

INPUTS = normalize(INPUTS);

partition='Kfold'
k=10

CV=cvpartition(OUTPUTS,partition,k);
for n=1:3
    for i=1:k
        trIdx = CV.training(i);
        MdL{i} = fitcsvm(INPUTS(trIdx,:),OUTPUTS(trIdx,:),args{:});
    end
end

for j=1:k
    trIdx = CV.training(j);
    y_pred = predict(MdL{j},INPUTS(trIdx,:));
    [CM, ORDERCM] = confusionmat(OUTPUTS(trIdx,:), y_pred);
    for i=1:Numclass
        [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
    end
end
trainACC = mean(ACC);

for j=1:k
    teIdx = CV.test(j);
    y_pred = predict(MdL{j},INPUTS(teIdx,:));
    [CM, ORDERCM] = confusionmat(OUTPUTS(teIdx,:), y_pred);
    for i=1:Numclass
        [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
    end
end
testACC = mean(ACC);
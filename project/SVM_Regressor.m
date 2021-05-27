function [trainACC,testACC] = SVM_Regressor(dataset,windowSize,args)
arguments
    dataset(1,:) char;
    windowSize(1,1) double;
    args(1,:) cell;
end

load(dataset);

[INPUTS,OUTPUTS] = slidingwindow(x(1500:3000),windowSize);
INPUTS = normalize(INPUTS);

partition='Kfold'
k=10
trainACC = []
testACC = []
CV=cvpartition(OUTPUTS,partition,k,'Stratify',false);
for n=1:3
    for i=1:k
        trIdx = CV.training(i);
        MdL{i} = fitrsvm(INPUTS(trIdx,:),OUTPUTS(trIdx,:),args{:});
    end


    for j=1:k
        trIdx=CV.training(j);
        y_pred = predict(MdL{j},INPUTS(trIdx,:));
        trainACC(end+1) = mean((OUTPUTS(trIdx,:)- y_pred).^2);
    end
  

    for j=1:k
        teIdx = CV.test(j);
        y_pred = predict(MdL{j},INPUTS(teIdx,:));
        testACC(end+1) = mean((OUTPUTS(teIdx,:)- y_pred).^2)
        
    end
end

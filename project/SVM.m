function [trainACC,testACC] = SVM(dataset,PredictorNames,ResponseName,args)
arguments
    dataset(1,:) char;
    PredictorNames(1,:) cell;
    ResponseName(1,:) char;
    args(1,:) cell;
end

clear MdL;
load(dataset);
trainACC = []
testACC = []

INPUTS = normalize(INPUTS);

partition='Kfold'
k=10

CV=cvpartition(OUTPUTS,partition,k);
for n=1:3
    for i=1:k
        trIdx = CV.training(i);
        MdL{i} = fitcsvm(INPUTS(trIdx,:),OUTPUTS(trIdx,:),args{:});
    end


    for j=1:k
        trIdx = CV.training(j);
        y_pred = predict(MdL{j},INPUTS(trIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(trIdx,:), y_pred);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        trainACC(end+1) = mean(ACC(:,j));
    end
  

    for j=1:k
        teIdx = CV.test(j);
        y_pred = predict(MdL{j},INPUTS(teIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(teIdx,:), y_pred);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        testACC(end+1) = mean(ACC(:,j));
    end
    
    for i=1:Numclass
        fprintf('Resultados para la clase %s\n', ORDERCM(i));
        fprintf('\t Recall %3.2f\n', mean(Recall(i,:)));
        fprintf('\t Precision %3.2f\n', mean(Precision(i,:)));
        fprintf('\t Recall %3.2f\n', mean(Recall(i,:)));
        fprintf('\t Valor Predictivo Negativo %3.2f\n', mean(VPN(i,:)));
        fprintf('\t Accuracy %3.2f\n', mean(ACC(i,:)));
        fprintf('\t F1Score %3.2f\n', mean(F1Score(i,:)));
    end

    fprintf('\nLa media del Accuracy del conjunto de test es %3.2f', mean(mean(ACC)));
    fprintf('\nEl F1Score GLOBAL es %3.2f', mean(mean(F1Score)))
    
end

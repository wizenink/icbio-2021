function [trainACC,testACC] = ANN(dataset,PredictorNames,ResponseName,hiddenLayerSize,trainMethod,activationFunction,metric)
arguments
    dataset(1,:) char;
    PredictorNames(1,:) cell;
    ResponseName(1,:) char;
    hiddenLayerSize(1,1) double = 8;
    trainMethod(1,:) char = 'trainscg';
    activationFunction(1,:) char = 'tansig';
    metric(1,:) char = 'crossentropy';
    
    
end

load(dataset);
OUTPUTS = grp2idx(OUTPUTS);
TARGETS = OUTPUTS;

TARGETS = [TARGETS,TARGETS(:,1)]; %one-hot vector
TARGETS(find(TARGETS(:, 1)== 1),3) = 0;
TARGETS(find(TARGETS(:, 1)== 1),2) = 0;
TARGETS(find(TARGETS(:, 1)== 1),1) = 1;

TARGETS(find(TARGETS(:, 1)== 2),3) = 0;
TARGETS(find(TARGETS(:, 1)== 2),2) = 1;
TARGETS(find(TARGETS(:, 1)== 2),1) = 0;

TARGETS(find(TARGETS(:, 1)== 3),3) = 1;
TARGETS(find(TARGETS(:, 1)== 3),2) = 0;
TARGETS(find(TARGETS(:, 1)== 3),1) = 0;
INPUTS = INPUTS'; TARGETS = TARGETS';


partition='Kfold'
k=10

CV=cvpartition(OUTPUTS,partition,k);

trainACC = []
testACC = []


for n=1:3
    net = patternnet(hiddenLayerSize,trainMethod,metric);
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0;
    net.trainParam.epochs = 1000;
    net.trainParam.time = inf;
    net.trainParam.max_fail = 6;%Early Stopping
    net.trainParam.min_grad = 1.0e-7;%Delta gradient for early stopping
    
    net.layers{1}.transferFcn = activationFunction;
    net.layers{2}.transferFcn = activationFunction;
    net.trainParam.showWindow = true;

    if(strcmp(trainMethod, 'trainscg'))
            net.trainParam.sigma = 5.0e-5; % Gradiente conjugado escalado: determina cambio de pesos para la segunda derivada
            net.trainParam.lambda = 5.0e-7; %Gradiente conjugado escalado: parámetro para regular la indefinición de la matriz Hessiana
        else
            net.trainParam.mu = 1.0e-03; %  Levenberg Marquardt: valor inicial del parámetro mu
            net.trainParam.mu_dec = 0.1; %  Levenberg Marquardt: decremento de mu
    end

    init(net);
    
    for i=1:k
        trIdx = CV.training(i);
        [net, tr] = train(net,INPUTS(:,trIdx), TARGETS(:,trIdx));
    end

    for j=1:k
        trIdx = CV.training(j);
        y_pred = net(INPUTS(:,trIdx));
        y_real = TARGETS(:,trIdx);
        
        %e = gsubtract(y_pred,y_real);
        performance = perform(net,y_pred,y_real);
        pred_train = vec2ind(y_pred);
        y_train = vec2ind(y_real);
        [CM, ORDERCM] = confusionmat(pred_train,y_train);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        trainACC(end+1) = mean(ACC(:,j));
    end
    %trainACC = mean(ACC);
    

    for j=1:k
        teIdx = CV.test(j);
        y_pred = net(INPUTS(:,teIdx));
        y_real = TARGETS(:,teIdx);
        performance = perform(net,y_pred,y_real);
        pred_test = vec2ind(y_pred);
        y_test = vec2ind(y_real);
        [CM, ORDERCM] = confusionmat(pred_test,y_test);
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
end
%testACC = mean(ACC);
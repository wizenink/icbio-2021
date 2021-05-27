function [trainACC,testACC] = ANN_Regressor(dataset,windowSize,hiddenLayerSize,trainMethod,activationFunction,metric)
arguments
    dataset(1,:) char;
    windowSize(1,1) double;
    hiddenLayerSize(1,1) double = 8;
    trainMethod(1,:) char = 'trainscg';
    activationFunction(1,:) char = 'tansig';
    metric(1,:) char = 'crossentropy';
    
    
end
load(dataset);


[INPUTS, OUTPUTS] = slidingwindow(x(1500:3000), windowSize);
INPUTS = normalize(INPUTS);
OUTPUTS = normalize(OUTPUTS);
INPUTS = INPUTS'; 
OUTPUTS = OUTPUTS';

partition='Kfold'
k=10



trainACC = [];
testACC = [];


for n=1:3
    CV=cvpartition(OUTPUTS,partition,k,'Stratify',false);
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
        [net, tr] = train(net,INPUTS(:,trIdx), OUTPUTS(:,trIdx));
    end

    for j=1:k
        trIdx = CV.training(j);
        y_pred = net(INPUTS(:,trIdx));
        trainACC(end+1) = mean((OUTPUTS(:, trIdx)- y_pred).^2);
        
    end
    %trainACC = mean(ACC);
    

    for j=1:k
        teIdx = CV.test(j);
        y_pred = net(INPUTS(:,teIdx));
        testACC(end+1) = mean((OUTPUTS(:,teIdx)- y_pred).^2);
    end
end
%testACC = mean(ACC);
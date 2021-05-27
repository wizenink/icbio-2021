function [trainACC,testACC] = Tree(dataset,PredictorNames,ResponseName,MinLeafSize,MinParentSize,debug)
arguments
    dataset(1,:) char;
    PredictorNames(1,:) cell;
    ResponseName(1,:) char;
    MinLeafSize(1,1) double = 1; 
    MinParentSize(1,1) double = 10;
    debug(1,1) logical = false;
end

%%Funcion de tree
%clear all;
load(dataset)

%tree = fitctree(x, y, nombre, valor)
%Variables: minnumsplit, Minleafsize, MinParentsize
 %Carga del WS
rng('shuffle'); %Semilla de aleatoriedad
trainACC = []
testACC = []

%% Árbol de decisión v1:
MaxNumSplits = size(INPUTS, 1) -1; % valor por defecto  n · muestras - 1 (n tamaño del conjunto de entrenamiento). 
%MinLeafSize = 1; % valor por defecto 1. 
%MinParentSize = 10; % valor por defecto 10. 
%ResponseName = 'Iris Type';
%PredictorNames = {'Sepal Lenght', 'Sepal Widht', 'Petal Lenght', 'Petal Width'};
%%CONDICIONES DE EXPERIMENTACIO PARA ARBOLES%%
TipoCV = 'KFold'; %LeaveOut
k = 10; %Numero de modelos
DiscrimType = 'Arbol de decision'
%%Conjunto de entrenamientos
CV = cvpartition(OUTPUTS, TipoCV, k)
for i = 1:3
    for i = 1:k
        trIdx = CV.training(i); %Extracción de los indices del conjunto de entrenamiento
        Mdl{i} = fitctree(INPUTS(trIdx,:), OUTPUTS(trIdx,:), 'MaxNumSplits', MaxNumSplits,'MinLeafSize', MinLeafSize,'MinParentSize', MinParentSize,'PredictorNames',PredictorNames,'ResponseName',ResponseName);
        if debug == true
            view(Mdl{i}, 'Mode', 'graph') %Para sacarlo solo con texto e imagen
        end
    end

    %Cálculo medidas de rendimiento para los conjuntos de training
    for j = 1:k
        REALTRAINOUTPUTS = predict(Mdl{j}, INPUTS(trIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(trIdx,:), REALTRAINOUTPUTS);

        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        trainACC(end+1) = mean(ACC(:,j));
    end
    results(1,:) = mean(ACC)
    %Impresion por pantalla conjunto de training
    for i=1:Numclass
        fprintf('Resultados para la clase %s\n', ORDERCM(i));
        fprintf('\t Recall %3.2f\n', mean(Recall(i,:)));
        fprintf('\t Precision %3.2f\n', mean(Precision(i,:)));
        fprintf('\t Valor Predictivo Negativo %3.2f\n', mean(VPN(i,:)));
        fprintf('\t Especificidad %3.2f\n', mean(Spec(i,:)))
        fprintf('\t Accuracy %3.2f\n', mean(ACC(i,:)));
        fprintf('\t F1Score %3.2f\n', mean(F1Score(i,:)));
    end

    fprintf('\nLa accuracy GLOBAL es %3.2f\n', mean(mean(ACC)))
    %Training
    for j = 1:k
        teIdx = CV.test(j);
        REALTESTOUTPUTS = predict(Mdl{j}, INPUTS(teIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(teIdx,:), REALTESTOUTPUTS);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        matriz_a1 = [mean(ACC)]
        resultados = [mean(Recall); mean(Spec);mean(Precision); mean(VPN); mean(ACC); mean(F1Score)]
        testACC(end+1) = mean(ACC(:,j));
    end
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
fprintf('\nEl F1Score GLOBAL es %3.2f', mean(mean(F1Score)));
%save('ACC_arbol_1', 'matriz_a1')
%save('Resultados_arbol_1', 'resultados')



function [trainACC,testACC] = lineal(dataset)
disp('Empezando EL PROCESO DE ENTRENAMIENTO para Cancer (con CV.partition)');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

%clear all;
load(dataset); %Carga del WS
rng('shuffle'); %Semilla de aleatoriedad
trainACC = []
testACC = []

%%CONDICIONES DE EXPERIMENTACIO PARA EL DISCRIMINANTE LINEAL%%
DiscrimType = 'linear';
TipoCV = 'KFold'; %LeaveOut
k = 10; %Numero de modelos

%Ejercicio 1: entrenar utilizando 10-fold a trav�s del cvpartition
CV = cvpartition(OUTPUTS, TipoCV, k); %CV.traing/test (i) nos da los diferentes conjuntos de entrenamiento/test

for i = 1:3
    %Bucle para entrenar los diferentes modelos (10) Conjuntos de entrenamientos y
    %obtenemos los modelos
    for i = 1:k
        trIdx = CV.training(i); %Extracci�n de los indices del conjunto de entrenamiento
        Mdl{i} = fitcdiscr(INPUTS(trIdx,:), OUTPUTS(trIdx,:), 'DiscrimType', DiscrimType);
    end

    %%%% CALCULAMOS Y MOSTRAMOS LOS RESULTADOS medios de los 10 modelos
    fprintf('Resultados de los Trainings para el Cancer para un disciminante tipo %s\n', DiscrimType);
    %Bucle para calcular las confusion matrix y las medidas de rendimiento de
    %cada modelo del conjunto de entrenamiento. Obtenemos la media de los
    %modelos para cada clase. (Numero de clases x Nmodelos) 
    for j = 1:k
        REALTRAINOUTPUTS = predict(Mdl{j}, INPUTS(trIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(trIdx,:), REALTRAINOUTPUTS);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        trainACC(end+1) = mean(ACC(:,j));
    end

    %IMPRESION DE RESULTADOS de cada clase como la media de los 10
    %entrenamientos
    for i=1:Numclass
        fprintf('Resultados para la clase %s\n', ORDERCM(i));
        fprintf('\t Recall %3.2f\n', mean(Recall(i,:)));
        fprintf('\t Precision %3.2f\n', mean(Precision(i,:)));
        fprintf('\t Spec %3.2f\n', mean(Spec(i,:)));
        fprintf('\t Valor Predictivo Negativo %3.2f\n', mean(VPN(i,:)));
        fprintf('\t Accuracy %3.2f\n', mean(ACC(i,:)));
        fprintf('\t F1Score %3.2f\n', mean(F1Score(i,:)));
    end

    fprintf('\nLa accuracy GLOBAL es %3.2f\n', mean(mean(ACC)))

    
    %BUCLE para los tests
    fprintf('Resultados de Test para el Cancer para un disciminante tipo %s\n', DiscrimType);
    for j = 1:k
        teIdx = CV.test(j);
        REALTESTOUTPUTS = predict(Mdl{j}, INPUTS(teIdx,:));
        [CM, ORDERCM] = confusionmat(OUTPUTS(teIdx,:), REALTESTOUTPUTS);
        for i=1:Numclass
            [Recall(i,j),Spec(i,j),Precision(i,j),VPN(i,j),ACC(i,j),F1Score(i,j)] = performanceIndexes(CM,i);
        end
        testACC(end+1) = mean(ACC(:,j));
    end

    %Impresion de los test
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

    
    %%Calculo del discriminante final utilizando todos los ejemplos%%%

end
%Guardar los resultados del test
savefile = 'ResultadosLinealCancer';
%save(savefile, 'Mdl', 'Recall', 'Spec', 'Precision', 'VPN', 'ACC', 'F1Score');

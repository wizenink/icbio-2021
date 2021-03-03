clear all;

rng('shuffle');
load fisheriris;


DiscrimType = 'linear';

%Entrenamiento:
Mdl = fitcdiscr(meas,species, 'DiscrimType',DiscrimType);


y_pred = predict(Mdl,meas); 

confusionmat(species, y_pred);

[confusion_matrix, cm_order] = confusionmat(species, y_pred);

fprintf('Confusion Matrix for %s discriminant\n', DiscrimType);
fprintf('Class order: \n');
fprintf('\t - %s \n \t - %s \n \t - %s \n', cm_order{1}, cm_order{2}, cm_order{3});

fprintf('Confusion Matrix: \n');
disp(confusion_matrix);

VP=confusion_matrix(1,1);
VN=confusion_matrix(2,2)+confusion_matrix(2,3)+confusion_matrix(3,2)+confusion_matrix(3,3);
FP=confusion_matrix(2,1)+confusion_matrix(3,1);
FN=confusion_matrix(1,2)+confusion_matrix(1,3);

EX=(VP+VN)/(VP+VN+FP+FN); %Exactitud
S=VP/(VP+FN);     %Sensibilidad
P=VP/(VP+FP);    %Precision
VPN=VN/(VN+FN);  %Valor predictivo negativo
ES=VN/(VN+FP);   %Especificidad
F1SCORE=2*((P*S)/(P+S));

%Salida de los datos
fprintf('Medidas para SETOSA: \n')
fprintf('\t ·Exactitud: %f \n', EX)
fprintf('\t ·Sensibilidad: %f \n', S)
fprintf('\t ·Precision: %f \n', P)
fprintf('\t ·VPN: %f \n', VPN)
fprintf('\t ·Especificidad: %f \n', ES)
fprintf('\t ·F1SCORE: %f \n', F1SCORE)

clear all;
cancerVars = {'VarName1','VarName2',  'VarName3', 'VarName4', 'VarName5', 'VarName6', 'VarName7', 'VarName8', 'VarName9'};
irisVars = {'Sepal Lenght', 'Sepal Widht', 'Petal Lenght', 'Petal Width'};
%% Cancer section

results = zeros(5,10);
etiquetas = ['Discrim  lineal';'Discrim  quadrt';'Decision Tree 1';'Decision Tree 2';'Decision Tree 3'];

[lineal_train,lineal_test] = lineal('cancerWS.mat');
results(1,:) = lineal_test;
[quadratic_tr,quadratic_test] = quadratic('cancerWS.mat');
results(2,:) = quadratic_test;

[tree1_train,tree1_test] = Tree('cancerWS.mat',cancerVars,'Cancer');
results(3,:) = tree1_test;

[tree2_train,tree2_test] = Tree('cancerWS.mat',cancerVars,'Cancer',5,10);
results(4,:) = tree2_test;

[tree3_train,tree3_test] = Tree('cancerWS.mat',cancerVars,'Cancer',2,5);
results(5,:) = tree3_test;


%pvalue = testEstadistico(transpose(results), etiquetas);


% Iris section

results = zeros(5,10);
etiquetas = ['Discrim  lineal';'Discrim  quadrt';'Decision Tree 1';'Decision Tree 2';'Decision Tree 3'];

[lineal_train,lineal_test] = lineal('irisWS.mat');
results(1,:) = lineal_test;
[quadratic_tr,quadratic_test] = quadratic('irisWS.mat');
results(2,:) = quadratic_test;

[tree1_train,tree1_test] = Tree('irisWS.mat',irisVars,'Iris Type');
results(3,:) = tree1_test;

[tree2_train,tree2_test] = Tree('irisWS.mat',irisVars,'Iris Type',5,10);
results(4,:) = tree2_test;

[tree3_train,tree3_test] = Tree('irisWS.mat',irisVars,'Iris Type',2,5);
results(5,:) = tree3_test;


pvalue = testEstadistico(transpose(results), etiquetas);
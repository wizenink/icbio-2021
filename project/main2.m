clear all;
cancerVars = {'VarName1','VarName2',  'VarName3', 'VarName4', 'VarName5', 'VarName6', 'VarName7', 'VarName8', 'VarName9'};
irisVars = {'Sepal Lenght', 'Sepal Widht', 'Petal Lenght', 'Petal Width'};
%% Cancer section

results = zeros(2,10);
etiquetas = ['SVM1';'SVM2'];
args = {'KernelFunction', 'polynomial'};

[svm1_train,svm1_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
results(1,:) = svm1_test;

[svm2_train,svm2_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
results(2,:) = svm2_test;


pvalue = testEstadistico(transpose(results), etiquetas);


% Iris section

results = zeros(2,10);
etiquetas = ['SVM1';'SVM2'];

[svm1_test,svm1_test] = SVM_Multiclass('irisWS.mat',irisVars,'Iris Type',args);
results(1,:) = svm1_test;

%[tree3_train,tree3_test] = Tree('irisWS.mat',irisVars,'Iris Type');
%results(5,:) = tree3_test;


%pvalue = testEstadistico(transpose(results), etiquetas);
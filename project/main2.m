clear all;
cancerVars = {'VarName1','VarName2',  'VarName3', 'VarName4', 'VarName5', 'VarName6', 'VarName7', 'VarName8', 'VarName9'};
irisVars = {'Sepal Lenght', 'Sepal Widht', 'Petal Lenght', 'Petal Width'};
biodegVars = {};
for n=1:41
    biodegVars{end+1} = sprintf('VarName%d',n);
end
biodegVars{end+1} = 'Response'

% Iris section 1

results = zeros(15,30);

etiquetas = ['LIND';'QUAD';'TR01';'TR02';'TR03';'SVM1';'SVM2';'SVM3';'SVM4';'ANN1';'ANN2';'ANN3';'ANN4';'ANN5';'ANN6'];


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

args = {'KernelFunction', 'polynomial'};
[svm1_train,svm1_test] = SVM_Multiclass('irisWS.mat',irisVars,'Iris Type',args,'onevsone');
results(6,:) = svm1_test;

args = {'KernelFunction', 'polynomial','PolynomialOrder',5};
[svm2_train,svm2_test] = SVM_Multiclass('irisWS.mat',irisVars,'Iris Type',args);
results(7,:) = svm2_test;

args = {'KernelFunction', 'polynomial','PolynomialOrder',6};
[svm3_train,svm3_test] = SVM_Multiclass('irisWS.mat',irisVars,'Iris Type',args);
results(8,:) = svm3_test;

args = {'KernelFunction', 'rbf'};
[svm4_train,svm4_test] = SVM_Multiclass('irisWS.mat',irisVars,'Iris Type',args);
results(9,:) = svm4_test;

[ann1_train, ann1_test] = ANN('irisWS.mat',irisVars,'Iris Type',4,'trainscg');
results(10,:) = ann1_test;

[ann2_train, ann2_test] = ANN('irisWS.mat',irisVars,'Iris Type',16,'trainscg');
results(11,:) = ann2_test;

[ann3_train, ann3_test] = ANN('irisWS.mat',irisVars,'Iris Type',12,'trainscg','purelin');
results(12,:) = ann3_test;


[ann4_train, ann4_test] = ANN('irisWS.mat',irisVars,'Iris Type',4,'trainlm');
results(13,:) = ann4_test;

[ann5_train, ann5_test] = ANN('irisWS.mat',irisVars,'Iris Type',16,'trainlm');
results(14,:) = ann5_test;

[ann6_train, ann6_test] = ANN('irisWS.mat',irisVars,'Iris Type',12,'trainlm','purelin');
results(15,:) = ann6_test;

pvalue = testEstadistico(transpose(results), etiquetas);



% %% Cancer section 1
% 
% results = zeros(15,30);
% 
% etiquetas = ['LIND';'QUAD';'TR01';'TR02';'TR03';'SVM1';'SVM2';'SVM3';'SVM4';'ANN1';'ANN2';'ANN3';'ANN4';'ANN5';'ANN6'];
% [lineal_train,lineal_test] = lineal('cancerWS.mat');
% results(1,:) = lineal_test;
% [quadratic_tr,quadratic_test] = quadratic('cancerWS.mat');
% results(2,:) = quadratic_test;
% 
% [tree1_train,tree1_test] = Tree('cancerWS.mat',cancerVars,'Cancer');
% results(3,:) = tree1_test;
% 
% [tree2_train,tree2_test] = Tree('cancerWS.mat',cancerVars,'Cancer',5,10);
% results(4,:) = tree2_test;
% 
% [tree3_train,tree3_test] = Tree('cancerWS.mat',cancerVars,'Cancer',2,5);
% results(5,:) = tree3_test;
% args = {'KernelFunction', 'polynomial'};
% [svm1_train,svm1_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
% results(6,:) = svm1_test;
% 
% args = {'KernelFunction', 'polynomial','PolynomialOrder',5};
% [svm2_train,svm2_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
% results(7,:) = svm2_test;
% 
% args = {'KernelFunction', 'polynomial','PolynomialOrder',6};
% [svm3_train,svm3_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
% results(8,:) = svm3_test;
% 
% args = {'KernelFunction', 'rbf'};
% [svm4_train,svm4_test] = SVM('cancerWS.mat',cancerVars,'Cancer',args);
% results(9,:) = svm4_test;
% 
% [ann1_train, ann1_test] = ANN('cancerWS.mat',cancerVars,'Cancer',4,'trainscg');
% results(10,:) = ann1_test;
% 
% [ann2_train, ann2_test] = ANN('cancerWS.mat',cancerVars,'Cancer',16,'trainscg');
% results(11,:) = ann2_test;
% 
% [ann3_train, ann3_test] = ANN('cancerWS.mat',cancerVars,'Cancer',12,'trainscg','purelin');
% results(12,:) = ann3_test;
% 
% 
% [ann4_train, ann4_test] = ANN('cancerWS.mat',cancerVars,'Cancer',4,'trainlm');
% results(13,:) = ann4_test;
% 
% [ann5_train, ann5_test] = ANN('cancerWS.mat',cancerVars,'Cancer',16,'trainlm');
% results(14,:) = ann5_test;
% 
% [ann6_train, ann6_test] = ANN('cancerWS.mat',cancerVars,'Cancer',12,'trainlm','purelin');
% results(15,:) = ann6_test;
% 
% pvalue = testEstadistico(transpose(results), etiquetas);



% % %% Biodeg section 1
% % 
% results = zeros(15,30);
% 
% 
% etiquetas = ['LIND';'QUAD';'TR01';'TR02';'TR03';'SVM1';'SVM2';'SVM3';'SVM4';'ANN1';'ANN2';'ANN3';'ANN4';'ANN5';'ANN6'];
% [lineal_train,lineal_test] = lineal('biodegWS.mat');
% results(1,:) = lineal_test;
% 
% 
% [quadratic_tr,quadratic_test] = quadratic('biodegWS.mat');
% results(2,:) = quadratic_test;
% 
% [tree1_train,tree1_test] = Tree('biodegWS.mat',biodegVars,'Biodeg');
% results(3,:) = tree1_test;
% 
% [tree2_train,tree2_test] = Tree('biodegWS.mat',biodegVars,'Biodeg',5,10);
% results(4,:) = tree2_test;
% 
% [tree3_train,tree3_test] = Tree('biodegWS.mat',biodegVars,'Biodeg',2,5);
% results(5,:) = tree3_test;
% 
% args = {'KernelFunction', 'polynomial'};
% [svm1_train,svm1_test] = SVM('biodegWS.mat',biodegVars,'Biodeg',args);
% results(6,:) = svm1_test;
% 
% args = {'KernelFunction', 'polynomial','PolynomialOrder',5};
% [svm2_train,svm2_test] = SVM('biodegWS.mat',biodegVars,'Biodeg',args);
% results(7,:) = svm2_test;
% 
% args = {'KernelFunction', 'polynomial','PolynomialOrder',6};
% [svm3_train,svm3_test] = SVM('biodegWS.mat',biodegVars,'Biodeg',args);
% results(8,:) = svm3_test;
% 
% args = {'KernelFunction', 'rbf'};
% [svm4_train,svm4_test] = SVM('biodegWS.mat',biodegVars,'Biodeg',args);
% results(9,:) = svm4_test;
% 
% [ann1_train, ann1_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',24,'trainscg');
% results(10,:) = ann1_test;
% 
% [ann2_train, ann2_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',50,'trainscg');
% results(11,:) = ann2_test;
% 
% [ann3_train, ann3_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',50,'trainscg','purelin');
% results(12,:) = ann3_test;
% 
% 
% [ann4_train, ann4_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',21,'trainlm');
% results(13,:) = ann4_test;
% % 
% [ann5_train, ann5_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',42,'trainlm');
% results(14,:) = ann5_test;
% 
% [ann6_train, ann6_test] = ANN('biodegWS.mat',biodegVars,'Biodeg',30,'trainlm','purelin');
% results(15,:) = ann6_test;
% 
% pvalue = testEstadistico(transpose(results), etiquetas);




% %% Henon section 1
% 
% results = zeros(18,30);
% 
% etiquetas = ['SVM1';'SVM2';'SVM3';'SVM4';'SVM5';'SVM6';'ANN1';'ANN2';'ANN3';'ANN4';'ANN5';'ANN6';'ANN7';'ANN8';'ANN9';'ANNA';'ANNB';'ANNC';];
% args = {'KernelFunction', 'polynomial','BoxConstraint',10};
% [svm1_train,svm1_test] = SVM_Regressor('Henon.mat',10,args);
% results(1,:) = svm1_test;
%  
% args = {'KernelFunction', 'rbf','BoxConstraint',10,'KernelScale',2};
% [svm2_train,svm2_test] = SVM_Regressor('Henon.mat',10,args);
% results(2,:) = svm2_test;
% 
% args = {'KernelFunction', 'rbf','BoxConstraint',20,'KernelScale',2};
% [svm3_train,svm3_test] = SVM_Regressor('Henon.mat',10,args);
% results(3,:) = svm3_test;
% 
% args = {'KernelFunction', 'polynomial','BoxConstraint',10};
% [svm4_train,svm4_test] = SVM_Regressor('Henon.mat',20,args);
% results(4,:) = svm4_test;
% 
% 
% args = {'KernelFunction', 'rbf','BoxConstraint',10,'KernelScale',2};
% [svm5_train,svm5_test] = SVM_Regressor('Henon.mat',20,args);
% results(5,:) = svm5_test;
% 
% args = {'KernelFunction', 'rbf','BoxConstraint',20,'KernelScale',2};
% [svm6_train,svm6_test] = SVM_Regressor('Henon.mat',20,args);
% results(6,:) = svm6_test;
% 
% [ann1_train, ann1_test] = ANN_Regressor('Henon.mat',10,4,'trainscg','tansig','mse');
% results(7,:) = ann1_test;
% [ann2_train, ann2_test] = ANN_Regressor('Henon.mat',10,16,'trainscg','tansig','mse');
% results(8,:) = ann2_test;
% [ann3_train, ann3_test] = ANN_Regressor('Henon.mat',10,12,'trainscg','purelin','mse');
% results(9,:) = ann3_test;
% 
% [ann4_train, ann4_test] = ANN_Regressor('Henon.mat',20,4,'trainscg','tansig','mse');
% results(10,:) = ann4_test;
% [ann5_train, ann5_test] = ANN_Regressor('Henon.mat',20,16,'trainscg','tansig','mse');
% results(11,:) = ann5_test;
% [ann6_train, ann6_test] = ANN_Regressor('Henon.mat',20,12,'trainscg','purelin','mse');
% results(12,:) = ann6_test;
% 
% 
% [ann7_train, ann7_test] = ANN_Regressor('Henon.mat',10,4,'trainlm','tansig','mse');
% results(13,:) = ann7_test;
% [ann8_train, ann8_test] = ANN_Regressor('Henon.mat',10,16,'trainlm','tansig','mse');
% results(14,:) = ann8_test;
% [ann9_train, ann9_test] = ANN_Regressor('Henon.mat',10,12,'trainlm','purelin','mse');
% results(15,:) = ann9_test;
% 
% [ann10_train, ann10_test] = ANN_Regressor('Henon.mat',20,4,'trainlm','tansig','mse');
% results(16,:) = ann10_test;
% [ann11_train, ann11_test] = ANN_Regressor('Henon.mat',20,16,'trainlm','tansig','mse');
% results(17,:) = ann11_test;
% [ann12_train, ann12_test] = ANN_Regressor('Henon.mat',20,12,'trainlm','purelin','mse');
% results(18,:) = ann12_test;
% 
% 
% pvalue = testEstadistico(transpose(results), etiquetas);

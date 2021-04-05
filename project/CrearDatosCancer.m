%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /MATLAB Drive/breast-cancer-wisconsin-data.csv
%
% Auto-generated by MATLAB on 02-Apr-2021 12:24:25

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 11);


% Specify range and delimiter
opts.DataLines = [1, 699];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1","VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
breast_cancer_wisconsin_data = readtable("/MATLAB Drive/breast-cancer-wisconsin-data.csv", opts);
INPUTS = [breast_cancer_wisconsin_data.VarName2,breast_cancer_wisconsin_data.VarName3,breast_cancer_wisconsin_data.VarName4,breast_cancer_wisconsin_data.VarName5,breast_cancer_wisconsin_data.VarName6,breast_cancer_wisconsin_data.VarName7,breast_cancer_wisconsin_data.VarName8,breast_cancer_wisconsin_data.VarName9,breast_cancer_wisconsin_data.VarName10];
OUTPUTS = [breast_cancer_wisconsin_data.VarName11];
[Numclass, ~] = size(unique(OUTPUTS));
[Numdata,~] = size(INPUTS);
save('cancerWS','breast_cancer_wisconsin_data', 'INPUTS', 'OUTPUTS', 'Numclass', 'Numdata')


%% Clear temporary variables
clear opts
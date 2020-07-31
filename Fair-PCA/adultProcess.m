function [M, A, B, LA, LB] = adultProcess()

% preprocess the adult census data. The output of the function is the centered
% data as matrix M. Centered low educated group A and high educated as
% group B. 

addpath data/adult_census_income

data = csvread('adult_processed.csv', 1, 0);

% vector of sensitive attribute.
sensitive = data(:,9);

% income label.
label = data(:,14);
label_A = label(find(sensitive),:);
label_B = label(find(~sensitive),:);

% getting rid of the colum corresponding to the senstive attribute.
data(:,9) = []; 
data(:,13) = [];

n = size(data, 2);

% centering the data and normalizing the variance across each column
for i=1:n
   data(:,i) = data(:,i) - mean(data(:,i));
   data(:,i) = data(:,i)/std(data(:,i));
end

% data for low educated populattion
data_lowEd = data(find(sensitive),:);
lowEd_copy = data_lowEd;

% date for high educated population
data_highEd = data(find(~sensitive),:);
highEd_copy = data_highEd;

mean_lowEd = mean(lowEd_copy,1);
mean_highEd = mean(highEd_copy, 1);

% centering data for high- and low-educated
for i=1:n
   lowEd_copy(i,:) = lowEd_copy(i,:) - mean_lowEd;
end

for i=1:n
   highEd_copy(i,:) = highEd_copy(i,:) - mean_highEd;
end


M = data;
A = lowEd_copy;
B = highEd_copy;
LA = label_A;
LB = label_B;
    

end

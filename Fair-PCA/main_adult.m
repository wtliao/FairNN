clc;

[M, A, B, LA, LB] = adultProcess();

featureNum = 12;

coeff = pca(M);
coeff_A = pca(A);
coeff_B = pca(B);

loss_A = zeros(featureNum,1);
loss_B = zeros(featureNum,1);

z_last = zeros(featureNum, 1);
z = zeros(featureNum, 1);
lossFair_max = zeros(featureNum, 1);

lossFair_A = zeros(featureNum,1);
lossFair_B = zeros(featureNum,1);

% parameters of the mw algorithm
eta = 1;
T = 100; 


for ell=1:featureNum
    
    P = coeff(:,1:ell)*transpose(coeff(:,1:ell));
    
    approx_A = A*P;
    approx_B = B*P;
    
    % vanilla PCA's average loss on popultion A and B
    loss_A(ell) = loss(A, approx_A, ell)/size(A, 1);
    loss_B(ell) = loss(B, approx_B, ell)/size(B, 1);
    
    
    [P_fair,z(ell),P_last,z_last(ell)] = mw(A, B, ell,eta ,T);
    
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end
    
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1))-P_smart);
    
    approxFair_A = A*P_smart;
    approxFair_B = B*P_smart;
    
    lossFair_A = loss(A, approxFair_A, ell)/size(A, 1);
    lossFair_B = loss(B, approxFair_B, ell)/size(B, 1);
    lossFair_max(ell) = max([lossFair_A, lossFair_B]);
    
end

% classification using projected data
XC = [approx_A,ones(size(approx_A,1),1);approx_B,zeros(size(approx_B,1),1)];
X = real(XC);
scatter(approxFair_A(:,1),approxFair_A(:,4),1,'green')
hold on;
scatter(approxFair_B(:,1),approxFair_B(:,4),1,'blue')

Y = [LA;LB];

% train/test split
[train, test] = crossvalind('holdOut', Y, 0.5);
cp = classperf(Y);

% train&test
svmModel = fitcsvm(X(train,1:12),Y(train));
[preds, score] = predict(svmModel, X(test,1:12));

% predictive performance
CM = confusionmat(Y(test),preds);
TN = CM(1,1);
FP = CM(1,2);
FN = CM(2,1);
TP = CM(2,2);
Acc = (TP+TN)/(sum(sum(CM)));
BAcc = (TP/(FN+TP)+TN/(TN+FP))/2;

% get two groups
flag = X(test,13);
preds_A = preds(find(flag),:);
preds_B = preds(find(~flag),:);
test = Y(test);
YA = test(find(flag),:);
YB = test(find(~flag),:);

% predictive performance of female
CM_A = confusionmat(YA,preds_A);
TN_A = CM_A(1,1);
FP_A = CM_A(1,2);
FN_A = CM_A(2,1);
TP_A = CM_A(2,2);
TPR_A = TP_A/(TP_A+FN_A);
TNR_A = TN_A/(TN_A+FP_A);

% predictive performance of male
CM_B = confusionmat(YB,preds_B);
TN_B = CM_B(1,1);
FP_B = CM_B(1,2);
FN_B = CM_B(2,1);
TP_B = CM_B(2,2);
TPR_B = TP_B/(TP_B+FN_B);
TNR_B = TN_B/(TN_B+FN_B);

% fairness performance
EqOdds = abs(TPR_A-TPR_B)+abs(TNR_A-TNR_B);


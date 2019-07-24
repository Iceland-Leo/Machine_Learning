function [ model ] = MyLogisticNewtonTrain( traindata,trainlabel )
%MyLogisticNewtonTrain 牛顿法训练
maxIter = 1000;  %最大迭代次数
iter = 0;   %迭代计数器
[m,n] = size(traindata);
oldtheta = ones(n,1);  %初始化上次迭代theta的值
model.theta = zeros(n,1);  %初始化模型theta值

%当迭代次数小于最大迭代次数并且参数发生变化，则进行牛顿方法更新
while(iter < maxIter && ((oldtheta - model.theta)' * (oldtheta - model.theta) > 1e-15))
    oldtheta = model.theta;
    output = g(model,traindata);  %计算模型输出，即函数h的值
    grad = traindata' * (trainlabel - output);  %计算梯度
    Hessian = traindata' * diag(output) * diag(output - 1) * traindata; %计算Hessian矩阵
    model.theta = model.theta - Hessian \ grad;  %更新theta值
    iter = iter + 1
end


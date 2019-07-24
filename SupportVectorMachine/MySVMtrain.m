function [model] = MySVMtrain(traindata,trainlabel,C,epsilon,sigma)
%MySVMtrain,使用SMO进行算法求解

[trainNum,trainDimension] = size(traindata); %获取训练集样本个数以及维度
nonBoundList = []; %初始化在边界上的alpha下标值向量
model.alpha = zeros(trainNum,1); %初始化模型文件的alpha值向量
model.b = 0.0; %初始化偏置b
    
maxIter = 1000; %最大迭代次数
iter = 0; %迭代计数器
examineAll = 1; %检验所有数据的标志变量
numChanged = 0; %声明alpha值成对改变的计数器变量
kernel = computeRKernel(traindata,traindata,sigma); %计算RBF核函数值矩阵
%外层循环寻找第一个alpha
while ((iter <= maxIter) && ((numChanged > 0) || examineAll))
    numChanged = 0;
    if examineAll == 1
        for i = 1:length(model.alpha)
            [changed,model,nonBoundList] = innerLoop(i,trainlabel,nonBoundList,kernel,C,epsilon,model);
            numChanged = numChanged + changed;
        end
    else
        for i = 1:length(nonBoundList)
            [changed,model,nonBoundList] = innerLoop(i,trainlabel,nonBoundList,kernel,C,epsilon,model);
            numChanged = numChanged + changed;
        end
    end
    
    iter = iter + 1
    if examineAll == 1
        examineAll = 0;  %遍历一次所有数据点后,下一次循环只遍历边界上的alpha
    elseif numChanged == 0
        examineAll = 1; %若边界的alpha均满足KKT,则再遍历所有数据点
    end
end
end


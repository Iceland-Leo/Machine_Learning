function [ model ] = MyGDATrain(traindata,trainlabel);
%MyGDATrain 高斯判别分析模型训练

[m,n] = size(traindata);
index1 = find(trainlabel);   %找到正样本的下标值向量
index0 = find(~trainlabel);  %找到负样本的下标值向量
model.phi = sum(trainlabel) / m;   %计算p(y=1),即phi值
model.u0 = sum(traindata(index0,:)) / (m - sum(trainlabel));   %计算负样本高斯分布的均值u0
model.u1 = sum(traindata(index1,:)) / sum(trainlabel);   %计算正样本高斯分布的均值u1
model.sigma = zeros(n,n);    %初始化两个高斯分布的协方差矩阵
%计算协方差矩阵
for i = 1:m
    if trainlabel(i) == 0
        model.sigma = model.sigma + (traindata(i,:) - model.u0)' * (traindata(i,:) - model.u0);
    else
        model.sigma = model.sigma + (traindata(i,:) - model.u1)' * (traindata(i,:) - model.u1);
    end
end
model.sigma = model.sigma / m;

end


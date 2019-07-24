function [ model ] = MyMultinomialNBTrain( traindata,trainlabel )
%MyNBTrain 训练分类器
[m,n] = size(traindata); 
model.phiKY1 = zeros(1,n);    %初始化参数phi K|Y = 1
model.phiKY0 = zeros(1,n);    %初始化参数phi K|Y = 0
%分别得到y=1的数据和y=0的数据的下标值
index1 = find(trainlabel);
index0 = find(~trainlabel);

model.phiY = sum(trainlabel) / m;   %计算参数phiY
model.phiKY1 = (sum(traindata(index1,:)) + 1) / (sum(sum(traindata(index1,:))) + n);   %计算参数phi K|Y = 1，结果为行向量(考虑拉普拉斯平滑)
model.phiKY0 = (sum(traindata(index0,:)) + 1) / (sum(sum(traindata(index0,:))) + n);   %计算参数phi K|Y = 0，结果为行向量(考虑拉普拉斯平滑)
end


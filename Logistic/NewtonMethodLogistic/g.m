function [ output ] = g( model,traindata )
%g 计算模型输出
z = traindata * model.theta;
[m,n] = size(z);
output = zeros(m,1);
for i = 1:m
    output(i) = 1.0 / (1.0 + exp(-z(i)));
end
end


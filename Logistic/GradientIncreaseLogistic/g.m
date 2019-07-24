function [ output ] = g(model,traindata)
%g 计算模型输出
z = traindata * model.theta;
if z >= 0
    output = 1;
else
    output = 0;
end
end


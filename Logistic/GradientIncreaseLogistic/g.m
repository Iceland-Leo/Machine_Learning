function [ output ] = g(model,traindata)
%g ����ģ�����
z = traindata * model.theta;
if z >= 0
    output = 1;
else
    output = 0;
end
end


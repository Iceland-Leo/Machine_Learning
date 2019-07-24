function [ model ] = MyLogisticNewtonTrain( traindata,trainlabel )
%MyLogisticNewtonTrain ţ�ٷ�ѵ��
maxIter = 1000;  %����������
iter = 0;   %����������
[m,n] = size(traindata);
oldtheta = ones(n,1);  %��ʼ���ϴε���theta��ֵ
model.theta = zeros(n,1);  %��ʼ��ģ��thetaֵ

%����������С���������������Ҳ��������仯�������ţ�ٷ�������
while(iter < maxIter && ((oldtheta - model.theta)' * (oldtheta - model.theta) > 1e-15))
    oldtheta = model.theta;
    output = g(model,traindata);  %����ģ�������������h��ֵ
    grad = traindata' * (trainlabel - output);  %�����ݶ�
    Hessian = traindata' * diag(output) * diag(output - 1) * traindata; %����Hessian����
    model.theta = model.theta - Hessian \ grad;  %����thetaֵ
    iter = iter + 1
end


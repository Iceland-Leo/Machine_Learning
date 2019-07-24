 function [model] = MyLogisticTrain( traindata,trainlabel,alpha )
%MyLogisticTrain �ݶ�������ѵ��
maxIter = 1000;  %����������
iter = 0;   %����������
maxNotChange = 5;  %���в�����δ�ı��������
notChangeNum = 0;  %���в�����δ�ı�Ĵ���������
notChange = 0;  %���в�����δ�ı�ı�־��0��ʾδ�ı䣬1��ʾ�иı�
[m,n] = size(traindata);
model.theta = zeros(n,1);
while (iter < maxIter && notChangeNum < maxNotChange)
    %�ݶ�����
    for i = 1:n
        for j = 1:m
           oldtheta = model.theta(i);
           output = g(model,traindata(j,:));
           model.theta(i) = model.theta(i) + alpha * (trainlabel(j) - output) * traindata(j,i);  %�������ݶ���������
           if abs(model.theta(i) - oldtheta) > 1e-5
               notChange = 1;
           end
        end
    end
    
    if notChange == 0
        notChangeNum = notChangeNum + 1;
    end
        iter = iter + 1
end


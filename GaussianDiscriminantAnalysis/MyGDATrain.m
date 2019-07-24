function [ model ] = MyGDATrain(traindata,trainlabel);
%MyGDATrain ��˹�б����ģ��ѵ��

[m,n] = size(traindata);
index1 = find(trainlabel);   %�ҵ����������±�ֵ����
index0 = find(~trainlabel);  %�ҵ����������±�ֵ����
model.phi = sum(trainlabel) / m;   %����p(y=1),��phiֵ
model.u0 = sum(traindata(index0,:)) / (m - sum(trainlabel));   %���㸺������˹�ֲ��ľ�ֵu0
model.u1 = sum(traindata(index1,:)) / sum(trainlabel);   %������������˹�ֲ��ľ�ֵu1
model.sigma = zeros(n,n);    %��ʼ��������˹�ֲ���Э�������
%����Э�������
for i = 1:m
    if trainlabel(i) == 0
        model.sigma = model.sigma + (traindata(i,:) - model.u0)' * (traindata(i,:) - model.u0);
    else
        model.sigma = model.sigma + (traindata(i,:) - model.u1)' * (traindata(i,:) - model.u1);
    end
end
model.sigma = model.sigma / m;

end


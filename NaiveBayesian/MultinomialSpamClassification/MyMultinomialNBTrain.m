function [ model ] = MyMultinomialNBTrain( traindata,trainlabel )
%MyNBTrain ѵ��������
[m,n] = size(traindata); 
model.phiKY1 = zeros(1,n);    %��ʼ������phi K|Y = 1
model.phiKY0 = zeros(1,n);    %��ʼ������phi K|Y = 0
%�ֱ�õ�y=1�����ݺ�y=0�����ݵ��±�ֵ
index1 = find(trainlabel);
index0 = find(~trainlabel);

model.phiY = sum(trainlabel) / m;   %�������phiY
model.phiKY1 = (sum(traindata(index1,:)) + 1) / (sum(sum(traindata(index1,:))) + n);   %�������phi K|Y = 1�����Ϊ������(����������˹ƽ��)
model.phiKY0 = (sum(traindata(index0,:)) + 1) / (sum(sum(traindata(index0,:))) + n);   %�������phi K|Y = 0�����Ϊ������(����������˹ƽ��)
end


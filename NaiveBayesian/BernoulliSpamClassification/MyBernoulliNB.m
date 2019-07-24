tic;
close all;
clear;
clc;
format compact;

[traindata,trainlabel] = readData('MATRIX.TRAIN');   %��ȡѵ������
[m,n] = size(traindata);
%��Xij��Ϊ0������Ϊ1����ʾ����j�ڵ�i���ʼ��г���(���ڱ�ʾ���ֵĴ�����������ʾ����)����ͬ
for i = 1:m
    index = find(traindata(i,:));
    traindata(i,index) = 1;
end
[testdata,testlabel] = readData('MATRIX.TEST');    %��ȡ��������
[m,n] = size(testdata);
for i = 1:m
    index = find(testdata(i,:));
    testdata(i,index) = 1;
end

model = MyBernoulliNBTrain(traindata,trainlabel);  %��Ŭ��ģ�����ر�Ҷ˹������ѵ��
[ptest,accuracy] = MyBernoulliNBPredict(testdata,testlabel,model);   %ʹ�ò�Ŭ��ģ�����ر�Ҷ˹�������Բ��Լ�����Ԥ��
accuracy     %�������׼ȷ��
toc;
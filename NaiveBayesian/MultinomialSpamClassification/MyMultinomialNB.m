tic;
close all;
clear;
clc;
format compact;

[traindata,trainlabel] = readData('MATRIX.TRAIN');   %��ȡѵ������
[testdata,testlabel] = readData('MATRIX.TEST');    %��ȡ��������

model = MyMultinomialNBTrain(traindata,trainlabel);  %�����¼�ģ�����ر�Ҷ˹������ѵ��
[ptest,accuracy] = MyMultinomialNBPredict(testdata,testlabel,model);   %ʹ�ö����¼�ģ�����ر�Ҷ˹�������Բ��Լ�����Ԥ��
accuracy     %�������׼ȷ��
toc;
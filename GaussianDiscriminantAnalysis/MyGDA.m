tic;
close all;
clear;
clc;
format compact;
% ��������
AllTrainData = textread('datatraining.txt');
AllTestData1 = textread('datatest.txt');
AllTestData2 = textread('datatest2.txt');
traindata = AllTrainData(1:8143,1:5);  %��ȡ�������ݣ�����Բ��Լ�����ͬ����
traindata = zscore(traindata);  %�������滯������Բ��Լ�����ͬ����
trainlabel = AllTrainData(1:8143,6);  %��ȡ���ݷ����ǩ������Բ��Լ�����ͬ����

testdata1 = AllTestData1(1:2665,1:5);
testdata1 = zscore(testdata1);
testlabel1 = AllTestData1(1:2665,6);

testdata2 = AllTestData2(1:9752,1:5);
testdata2 = zscore(testdata2);
testlabel2 = AllTestData2(1:9752,6);

% ����ѵ�����Ͻ�������ģ��
model = MyGDATrain(traindata,trainlabel);  %��˹�б����ģ��ѵ��
% ���Լ��ϼ���
[ptest1,accuracy1] = MyGDAPredict(testdata1,testlabel1,model);  %�Բ��Լ�1Ԥ�⣬����Ԥ����������׼ȷ��
[ptest2,accuracy2] = MyGDAPredict(testdata2,testlabel2,model);  %�Բ��Լ�2Ԥ�⣬����Ԥ����������׼ȷ��
accuracy1     %��ʾ�Բ��Լ�1Ԥ���׼ȷ��
accuracy2     %��ʾ�Բ��Լ�2Ԥ���׼ȷ��
toc;
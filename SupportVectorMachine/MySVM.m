tic;
close all;
clear;
clc;
format compact;
% ������������
AllTrainData = textread('datatraining.txt');
AllTestData1 = textread('datatest.txt');
AllTestData2 = textread('datatest2.txt');
traindata = AllTrainData(1:8143,1:5);  %��ȡ�������ݣ�����Բ��Լ�����ͬ����
traindata = zscore(traindata);  %�������滯������Բ��Լ�����ͬ����
trainlabel = AllTrainData(1:8143,6);  %��ȡ���ݷ����ǩ������Բ��Լ�����ͬ����
%�޸ı�ǩֵ��ʹ����+1��-1����
for i = 1:8143
    if trainlabel(i) == 0
        trainlabel(i) = -1;
    end
end
testdata1 = AllTestData1(1:2665,1:5);
testdata1 = zscore(testdata1);
testlabel1 = AllTestData1(1:2665,6);
for i = 1:2665
    if testlabel1(i) == 0
        testlabel1(i) = -1;
    end
end
testdata2 = AllTestData2(1:9752,1:5);
testdata2 = zscore(testdata2);
testlabel2 = AllTestData2(1:9752,6);
for i = 1:9752
    if testlabel2(i) == 0
        testlabel2(i) = -1;
    end
end
% ����ѵ�����Ͻ�������ģ��
C = 0.12;  %�ͷ�����
sigma = 0.3;  %��˹(RBF)�˺�����sigmaֵ
epsilon = 0.001; %���龫�ȷ�Χ
model = MySVMtrain(traindata,trainlabel,C,epsilon,sigma);  %SMO�㷨ѵ��
% ���Լ��ϼ���
[ptest1,accuracy1] = MySVMpredict(traindata,trainlabel,testdata1,testlabel1,sigma,model);  %�Բ��Լ�1Ԥ�⣬����Ԥ����������׼ȷ��
[ptest2,accuracy2] = MySVMpredict(traindata,trainlabel,testdata2,testlabel2,sigma,model);  %�Բ��Լ�2Ԥ�⣬����Ԥ����������׼ȷ��
accuracy1     %��ʾ�Բ��Լ�1Ԥ���׼ȷ��
accuracy2     %��ʾ�Բ��Լ�2Ԥ���׼ȷ��
toc;
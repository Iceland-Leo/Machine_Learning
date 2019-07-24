tic;
close all;
clear;
clc;
format compact;
% ������������(Iris�β��)
%[attrib1,attrib2,attrib3,attrib4,class] = textread('IrisData.txt','%f%f%f%f%s','delimiter',',');
%attrib = [attrib1,attrib2,attrib3,attrib4];  %��������ֵ����
%label = zeros(150,1); %���ڴ��������ǩ
%���������
%label(strcmp(class,'Iris-setosa')) = 0; 
%label(strcmp(class,'Iris-versicolor')) = 1;
%label(strcmp(class,'Iris-virginica')) = 3;

%����ѵ��������֤����ȡÿ������������ǰ40����Ϊѵ��������10����Ϊ��֤��
%traindata = attrib([1:40 51:90],:);
%trainlabel = label([1:40 51:90],:);
%testdata = attrib([41:50 91:100],:);
%testlabel = label([41:50 91:100],:);

% ��������(�б������Ƿ�����)
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

%����ѵ������������ģ��
eta1 = 0.1;  %�����㵽������Ȩ��ѧϰ��
eta2 = 0.1;  %����㵽�������Ȩ��ѧϰ��
model = MyBPNNtrain(traindata,trainlabel,eta1,eta2);  %BP�㷨ѵ��

%���Լ��ϼ���
[ptest1,accuracy1] = MyBPNNpredict(testdata1,testlabel1,model);  %�Բ��Լ�Ԥ�⣬����Ԥ����������׼ȷ��
[ptest2,accuracy2] = MyBPNNpredict(testdata2,testlabel2,model);  %�Բ��Լ�Ԥ�⣬����Ԥ����������׼ȷ��
accuracy1     %��ʾ�Բ��Լ�Ԥ���׼ȷ��
accuracy2     %��ʾ�Բ��Լ�Ԥ���׼ȷ��
toc;
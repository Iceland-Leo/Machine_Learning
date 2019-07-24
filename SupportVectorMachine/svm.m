tic;
close all;
clear;
clc;
format compact;
% ������������
AllTrainData = textread('datatraining.txt');
AllTestData1 = textread('datatest.txt');
AllTestData2 = textread('datatest2.txt');
% ѡȡǰ200��������Ϊѵ�����ϣ���70��������Ϊ���Լ���
traindata = AllTrainData(1:8143,1:5);
traindata = zscore(traindata);
trainlabel = AllTrainData(1:8143,6);
%�޸ı�ǩֵ
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
model = svmtrain(trainlabel,traindata,'-s 0 -t 2 -c 0.1 -g 0.3');
% ����ģ��model����
model
Parameters = model.Parameters
Label = model.Label
nr_class = model.nr_class
totalSV = model.totalSV
nSV = model.nSV 
% Ԥ����Լ��ϱ�ǩ
[ptest1,acc1,b1] = svmpredict(testlabel1,testdata1,model);
[ptest2,acc2,b2] = svmpredict(testlabel2,testdata2,model);
toc;
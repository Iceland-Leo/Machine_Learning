tic;
close all;
clear;
clc;
format compact;
% 首先载入数据
AllTrainData = textread('datatraining.txt');
AllTestData1 = textread('datatest.txt');
AllTestData2 = textread('datatest2.txt');
traindata = AllTrainData(1:8143,1:5);  %获取样本数据，下面对测试集做相同处理
traindata = zscore(traindata);  %数据正规化，下面对测试集做相同处理
trainlabel = AllTrainData(1:8143,6);  %获取数据分类标签，下面对测试集做相同处理
%修改标签值，使得有+1、-1两类
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
% 利用训练集合建立分类模型
C = 0.12;  %惩罚因子
sigma = 0.3;  %高斯(RBF)核函数的sigma值
epsilon = 0.001; %检验精度范围
model = MySVMtrain(traindata,trainlabel,C,epsilon,sigma);  %SMO算法训练
% 测试集合检验
[ptest1,accuracy1] = MySVMpredict(traindata,trainlabel,testdata1,testlabel1,sigma,model);  %对测试集1预测，返回预测结果向量和准确率
[ptest2,accuracy2] = MySVMpredict(traindata,trainlabel,testdata2,testlabel2,sigma,model);  %对测试集2预测，返回预测结果向量和准确率
accuracy1     %显示对测试集1预测的准确率
accuracy2     %显示对测试集2预测的准确率
toc;
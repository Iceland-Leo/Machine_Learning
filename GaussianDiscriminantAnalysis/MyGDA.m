tic;
close all;
clear;
clc;
format compact;
% 载入数据
AllTrainData = textread('datatraining.txt');
AllTestData1 = textread('datatest.txt');
AllTestData2 = textread('datatest2.txt');
traindata = AllTrainData(1:8143,1:5);  %获取样本数据，下面对测试集做相同处理
traindata = zscore(traindata);  %数据正规化，下面对测试集做相同处理
trainlabel = AllTrainData(1:8143,6);  %获取数据分类标签，下面对测试集做相同处理

testdata1 = AllTestData1(1:2665,1:5);
testdata1 = zscore(testdata1);
testlabel1 = AllTestData1(1:2665,6);

testdata2 = AllTestData2(1:9752,1:5);
testdata2 = zscore(testdata2);
testlabel2 = AllTestData2(1:9752,6);

% 利用训练集合建立分类模型
model = MyGDATrain(traindata,trainlabel);  %高斯判别分析模型训练
% 测试集合检验
[ptest1,accuracy1] = MyGDAPredict(testdata1,testlabel1,model);  %对测试集1预测，返回预测结果向量和准确率
[ptest2,accuracy2] = MyGDAPredict(testdata2,testlabel2,model);  %对测试集2预测，返回预测结果向量和准确率
accuracy1     %显示对测试集1预测的准确率
accuracy2     %显示对测试集2预测的准确率
toc;
tic;
close all;
clear;
clc;
format compact;

[traindata,trainlabel] = readData('MATRIX.TRAIN');   %读取训练数据
[testdata,testlabel] = readData('MATRIX.TEST');    %读取测试数据

model = MyMultinomialNBTrain(traindata,trainlabel);  %多项事件模型朴素贝叶斯分类器训练
[ptest,accuracy] = MyMultinomialNBPredict(testdata,testlabel,model);   %使用多项事件模型朴素贝叶斯分类器对测试集进行预测
accuracy     %输出分类准确率
toc;
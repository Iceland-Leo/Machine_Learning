tic;
close all;
clear;
clc;
format compact;

[traindata,trainlabel] = readData('MATRIX.TRAIN');   %读取训练数据
[m,n] = size(traindata);
%将Xij不为0的项设为1，表示单词j在第i封邮件中出现(不在表示出现的次数，仅仅表示出现)，下同
for i = 1:m
    index = find(traindata(i,:));
    traindata(i,index) = 1;
end
[testdata,testlabel] = readData('MATRIX.TEST');    %读取测试数据
[m,n] = size(testdata);
for i = 1:m
    index = find(testdata(i,:));
    testdata(i,index) = 1;
end

model = MyBernoulliNBTrain(traindata,trainlabel);  %伯努利模型朴素贝叶斯分类器训练
[ptest,accuracy] = MyBernoulliNBPredict(testdata,testlabel,model);   %使用伯努利模型朴素贝叶斯分类器对测试集进行预测
accuracy     %输出分类准确率
toc;
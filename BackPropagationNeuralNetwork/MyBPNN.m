tic;
close all;
clear;
clc;
format compact;
% 首先载入数据(Iris鸢尾花)
%[attrib1,attrib2,attrib3,attrib4,class] = textread('IrisData.txt','%f%f%f%f%s','delimiter',',');
%attrib = [attrib1,attrib2,attrib3,attrib4];  %样本属性值矩阵
%label = zeros(150,1); %用于存放样本标签
%设置类别标号
%label(strcmp(class,'Iris-setosa')) = 0; 
%label(strcmp(class,'Iris-versicolor')) = 1;
%label(strcmp(class,'Iris-virginica')) = 3;

%划分训练集与验证集，取每种类型样本的前40个作为训练集，后10个作为验证集
%traindata = attrib([1:40 51:90],:);
%trainlabel = label([1:40 51:90],:);
%testdata = attrib([41:50 91:100],:);
%testlabel = label([41:50 91:100],:);

% 载入数据(判别屋内是否有人)
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

%利用训练集建立分类模型
eta1 = 0.1;  %隐含层到输出层的权重学习率
eta2 = 0.1;  %输入层到隐含层的权重学习率
model = MyBPNNtrain(traindata,trainlabel,eta1,eta2);  %BP算法训练

%测试集合检验
[ptest1,accuracy1] = MyBPNNpredict(testdata1,testlabel1,model);  %对测试集预测，返回预测结果向量和准确率
[ptest2,accuracy2] = MyBPNNpredict(testdata2,testlabel2,model);  %对测试集预测，返回预测结果向量和准确率
accuracy1     %显示对测试集预测的准确率
accuracy2     %显示对测试集预测的准确率
toc;
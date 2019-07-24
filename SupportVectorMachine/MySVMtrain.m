function [model] = MySVMtrain(traindata,trainlabel,C,epsilon,sigma)
%MySVMtrain,ʹ��SMO�����㷨���

[trainNum,trainDimension] = size(traindata); %��ȡѵ�������������Լ�ά��
nonBoundList = []; %��ʼ���ڱ߽��ϵ�alpha�±�ֵ����
model.alpha = zeros(trainNum,1); %��ʼ��ģ���ļ���alphaֵ����
model.b = 0.0; %��ʼ��ƫ��b
    
maxIter = 1000; %����������
iter = 0; %����������
examineAll = 1; %�����������ݵı�־����
numChanged = 0; %����alphaֵ�ɶԸı�ļ���������
kernel = computeRKernel(traindata,traindata,sigma); %����RBF�˺���ֵ����
%���ѭ��Ѱ�ҵ�һ��alpha
while ((iter <= maxIter) && ((numChanged > 0) || examineAll))
    numChanged = 0;
    if examineAll == 1
        for i = 1:length(model.alpha)
            [changed,model,nonBoundList] = innerLoop(i,trainlabel,nonBoundList,kernel,C,epsilon,model);
            numChanged = numChanged + changed;
        end
    else
        for i = 1:length(nonBoundList)
            [changed,model,nonBoundList] = innerLoop(i,trainlabel,nonBoundList,kernel,C,epsilon,model);
            numChanged = numChanged + changed;
        end
    end
    
    iter = iter + 1
    if examineAll == 1
        examineAll = 0;  %����һ���������ݵ��,��һ��ѭ��ֻ�����߽��ϵ�alpha
    elseif numChanged == 0
        examineAll = 1; %���߽��alpha������KKT,���ٱ����������ݵ�
    end
end
end


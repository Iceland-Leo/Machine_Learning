function [ model ] = MyBPNNtrain(traindata,trainlabel,eta1,eta2);
%ʹ�÷��򴫲��㷨�����������ѵ��������ģ���ļ�
[m,n] = size(traindata);%��ȡѵ�����ݸ���m������������ά��d

model.maxIter = 1000;%����������
model.notChangedNum = 5;%����������Ĵ������ޣ�����ֹͣ���������
model.eta1 = eta1;%���������㵽������Ȩ��ѧϰ��
model.eta2 = eta2;%��������㵽�������Ȩ��ѧϰ��
model.d = n;%���������������ά��d
model.l = 1;%��������������ά��l
model.q = floor(sqrt(model.d + model.l));%��ʼ���������񾭸�������������Ԫ������ sqrt(d+l)+a ȷ��������aΪ0-10����

model.v = rand(model.d,model.q);%��ʼ������㵽�������Ȩ��
model.r = rand(1,model.q);%��ʼ�����������ֵr
model.w = rand(model.q,model.l);%��ʼ�������㵽������Ȩ��
model.theta = rand();%��ʼ����������ֵtheta

Iter = 0;%��ǰ��������
notChanged = 0;%��ǰ����������Ĵ���

while Iter < model.maxIter && notChanged < model.notChangedNum
    changed = 0;%��־�Ƿ��в����仯
    for i = 1:m
        %���ݵ�ǰ�������㵱ǰ���������y
        alpha = compute(traindata(i,:),model.v);%��������㵽������ļ�Ȩ��
        b = f(alpha - model.r); %��������������b
        beta = compute(b,model.w);%���������㵽�����ļ�Ȩ��
        y = f(beta - model.theta);%�������������y
        %�����������Ԫ���ݶ���g
        g = y .* (1 - y) .* (trainlabel(i,:) - y);
        %������������Ԫ���ݶ���e
        e = b .* (1 - b) .* (g * model.w');
        %����Ȩ��w,v����ֵtheta,r
        deltaW = model.eta1 .* (b' * g);
        model.w = model.w + deltaW;%���������㵽������Ȩ��
        deltaTheta = -model.eta1 .* g;
        model.theta = model.theta + deltaTheta;%������������ֵ
        deltaV = model.eta2 .* (traindata(i,:)' * e);
        model.v = model.v + deltaV;%��������㵽�������Ȩ��
        deltaR = -model.eta2 .* e;
        model.r = model.r + deltaR;%�������������ֵ
        if length(find(deltaW)) ~= 0 || length(find(deltaTheta)) ~= 0 || length(find(deltaV)) ~= 0 || length(find(deltaR)) ~= 0
            changed = 1;
        end
    end
    
    if changed == 0
        notChanged = notChanged + 1;
    end
    Iter = Iter + 1
end
end
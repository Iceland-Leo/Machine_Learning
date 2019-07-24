function [ model ] = MyBPNNtrain(traindata,trainlabel,eta1,eta2);
%使用反向传播算法对神经网络进行训练，返回模型文件
[m,n] = size(traindata);%获取训练数据个数m，神经网络输入维度d

model.maxIter = 1000;%最大迭代次数
model.notChangedNum = 5;%参数均不变的次数上限，用于停止程序的条件
model.eta1 = eta1;%保存隐含层到输出层的权重学习率
model.eta2 = eta2;%保存输入层到隐含层的权重学习率
model.d = n;%保存神经网络的输入维度d
model.l = 1;%保存神经网络的输出维度l
model.q = floor(sqrt(model.d + model.l));%初始化隐含层神经个数，隐含层神经元个数由 sqrt(d+l)+a 确定，其中a为0-10的数

model.v = rand(model.d,model.q);%初始化输入层到隐含层的权重
model.r = rand(1,model.q);%初始化隐含层的阈值r
model.w = rand(model.q,model.l);%初始化隐含层到输出层的权重
model.theta = rand();%初始化输出层的阈值theta

Iter = 0;%当前迭代次数
notChanged = 0;%当前参数均不变的次数

while Iter < model.maxIter && notChanged < model.notChangedNum
    changed = 0;%标志是否有参数变化
    for i = 1:m
        %根据当前参数计算当前样本的输出y
        alpha = compute(traindata(i,:),model.v);%计算输入层到隐含层的加权和
        b = f(alpha - model.r); %计算隐含层的输出b
        beta = compute(b,model.w);%计算隐含层到输出层的加权和
        y = f(beta - model.theta);%计算输出层的输出y
        %计算输出层神经元的梯度项g
        g = y .* (1 - y) .* (trainlabel(i,:) - y);
        %计算隐含层神经元的梯度项e
        e = b .* (1 - b) .* (g * model.w');
        %更新权重w,v和阈值theta,r
        deltaW = model.eta1 .* (b' * g);
        model.w = model.w + deltaW;%更新隐含层到输出层的权重
        deltaTheta = -model.eta1 .* g;
        model.theta = model.theta + deltaTheta;%更新输出层的阈值
        deltaV = model.eta2 .* (traindata(i,:)' * e);
        model.v = model.v + deltaV;%更新输入层到隐含层的权重
        deltaR = -model.eta2 .* e;
        model.r = model.r + deltaR;%更新隐含层的阈值
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
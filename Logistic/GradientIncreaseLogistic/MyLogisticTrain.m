 function [model] = MyLogisticTrain( traindata,trainlabel,alpha )
%MyLogisticTrain 梯度上升法训练
maxIter = 1000;  %最大迭代次数
iter = 0;   %迭代计数器
maxNotChange = 5;  %所有参数均未改变的最大次数
notChangeNum = 0;  %所有参数均未改变的次数计数器
notChange = 0;  %所有参数均未改变的标志，0表示未改变，1表示有改变
[m,n] = size(traindata);
model.theta = zeros(n,1);
while (iter < maxIter && notChangeNum < maxNotChange)
    %梯度上升
    for i = 1:n
        for j = 1:m
           oldtheta = model.theta(i);
           output = g(model,traindata(j,:));
           model.theta(i) = model.theta(i) + alpha * (trainlabel(j) - output) * traindata(j,i);  %参数按梯度上升更新
           if abs(model.theta(i) - oldtheta) > 1e-5
               notChange = 1;
           end
        end
    end
    
    if notChange == 0
        notChangeNum = notChangeNum + 1;
    end
        iter = iter + 1
end


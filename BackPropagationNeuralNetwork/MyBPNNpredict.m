function [ptest,accuracy] = MyBPNNpredict(testdata,testlabel,model)
%使用训练得到的神经网络进行预测
[m,n] = size(testlabel);
ptest = zeros(m,n);
for i = 1:m
    alpha = compute(testdata(i,:),model.v);%计算输入层到隐含层的加权和
    b = f(alpha - model.r); %计算隐含层的输出b
    beta = compute(b,model.w);%计算隐含层到输出层的加权和
    ptest(i,:) = f(beta - model.theta);%计算输出层的输出y
end
ptest(find(ptest >= 0.5)) = 1;
ptest(find(ptest < 0.5)) = 0;
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end
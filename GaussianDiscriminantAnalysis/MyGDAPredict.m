function [ ptest,accuracy ] = MyGDAPredict(testdata,testlabel,model)
%MyGDAPredict 使用模型预测
[m,n] = size(testdata);
ptest = zeros(m,1);
for i = 1:m
    pos = exp(-((testdata(i,:) - model.u1) * inv(model.sigma) * (testdata(i,:) - model.u1)') / 2) / ((2*pi)^(n/2) * sqrt(det(model.sigma)));
    neg = exp(-((testdata(i,:) - model.u0) * inv(model.sigma) * (testdata(i,:) - model.u0)') / 2) / ((2*pi)^(n/2) * sqrt(det(model.sigma)));
    if pos >= neg
        ptest(i) = 1;
    end
end
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


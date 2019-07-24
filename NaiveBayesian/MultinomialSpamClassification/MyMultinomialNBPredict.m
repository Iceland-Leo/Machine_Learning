function [ ptest,accuracy ] = MyMultinomialNBPredict(testdata,testlabel,model)
%MyNBPredict 使用多项事件模型朴素贝叶斯分类对测试集进行预测
[m,n] = size(testdata);
ptest = zeros(m,1);
for i = 1:m
    index1 = find(testdata(i,:));   %找出数据中非0元的下标
    index0 = find(testdata(i,:) == 0);   %找出数据中0元下标
    %为了避免计算下溢，对结果取对数
    pos = log(model.phiY) + sum((1 - testdata(i,index0)) .* log(1 - model.phiKY1(index0))) + sum(testdata(i,index1) .* log(model.phiKY1(index1)));
    neg = log(1 - model.phiY) + sum((1 - testdata(i,index0)) .* log(1 - model.phiKY0(index0))) + sum(testdata(i,index1) .* log(model.phiKY0(index1)));
    
    if pos >= neg
        ptest(i) = 1;
    end
end
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


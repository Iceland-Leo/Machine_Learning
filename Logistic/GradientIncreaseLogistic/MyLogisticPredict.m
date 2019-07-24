function [ ptest,accuracy ] = MyLogisticPredict( testdata,testlabel,model )
%MyLogisticPredict ��logistic�ع�ģ�ͽ���Ԥ��
[m,n] = size(testdata);
ptest = zeros(m,1);
for i = 1:m
    ptest(i) = g(model,testdata(i,:));
end
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


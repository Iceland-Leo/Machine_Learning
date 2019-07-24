function [ ptest,accuracy ] =  MyLogisticPredict(testdata,testlabel,model)
%MyLogisticPredict ʹ��ţ�ٷ�������ģ��Ԥ��
[m,n] = size(testdata);
ptest = g(model,testdata);
for i = 1:m
    if ptest(i) >= 0.5
        ptest(i) = 1;
    else
        ptest(i) = 0;
    end
end
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


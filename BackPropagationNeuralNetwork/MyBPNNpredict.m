function [ptest,accuracy] = MyBPNNpredict(testdata,testlabel,model)
%ʹ��ѵ���õ������������Ԥ��
[m,n] = size(testlabel);
ptest = zeros(m,n);
for i = 1:m
    alpha = compute(testdata(i,:),model.v);%��������㵽������ļ�Ȩ��
    b = f(alpha - model.r); %��������������b
    beta = compute(b,model.w);%���������㵽�����ļ�Ȩ��
    ptest(i,:) = f(beta - model.theta);%�������������y
end
ptest(find(ptest >= 0.5)) = 1;
ptest(find(ptest < 0.5)) = 0;
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end
function [ ptest,accuracy ] = MyMultinomialNBPredict(testdata,testlabel,model)
%MyNBPredict ʹ�ö����¼�ģ�����ر�Ҷ˹����Բ��Լ�����Ԥ��
[m,n] = size(testdata);
ptest = zeros(m,1);
for i = 1:m
    index1 = find(testdata(i,:));   %�ҳ������з�0Ԫ���±�
    index0 = find(testdata(i,:) == 0);   %�ҳ�������0Ԫ�±�
    %Ϊ�˱���������磬�Խ��ȡ����
    pos = log(model.phiY) + sum((1 - testdata(i,index0)) .* log(1 - model.phiKY1(index0))) + sum(testdata(i,index1) .* log(model.phiKY1(index1)));
    neg = log(1 - model.phiY) + sum((1 - testdata(i,index0)) .* log(1 - model.phiKY0(index0))) + sum(testdata(i,index1) .* log(model.phiKY0(index1)));
    
    if pos >= neg
        ptest(i) = 1;
    end
end
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


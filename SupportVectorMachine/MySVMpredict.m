function [ptest,accuracy] = MySVMpredict(traindata,trainlabel,testdata,testlabel,sigma,model)
%MySVMpredict ʹ��ģ��Ԥ��
[m,n] = size(testdata);
ptest = ((model.alpha .* trainlabel)' * computeRKernel(traindata,testdata,sigma))';
ptest = sign(ptest);
temp = ptest - testlabel;
accuracy = sum(temp(:) == 0) / m;
end


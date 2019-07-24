function [kernel] = computeRKernel(data1,data2,sigma)
%computeKernel,º∆À„RBF∫À∫Ø ˝÷µ

[m1,n1] = size(data1);
[m2,n2] = size(data2);
kernel = zeros(m1,m2);
for i = 1:m1
    for j = 1:m2
        kernel(i,j) = exp(-power(norm(data1(i,:) - data2(j,:)),2) / (2 * sigma * sigma));
    end
end

end


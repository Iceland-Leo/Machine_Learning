function [ result ] = f(x)
%ʹ��sigmod������Ϊ����������㺯�����

result = 1 ./ (1 + exp(-x));    
end
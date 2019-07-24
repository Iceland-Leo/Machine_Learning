function [ result ] = f(x)
%使用sigmod函数作为激活函数，计算函数输出

result = 1 ./ (1 + exp(-x));    
end
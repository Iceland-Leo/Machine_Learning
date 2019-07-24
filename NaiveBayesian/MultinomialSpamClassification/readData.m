function [data,label] = readData(filename) 
%readData    ��ȡ����

fid = fopen(filename);  %���ļ�  
fgetl(fid);  %���������ļ���һ��(����) 

%��ȡ���ݼ���С�����ݼ�����Ϊm�����ݵ�ά��n(ʵ����nΪ���ʱ�ĵ�������)
mn = fscanf(fid, '%d %d\n', 2);    
m = mn(1);
n = mn(2);  

fgetl(fid);    %���������ļ�������(���ʱ�ʵ������һ���ܳ���һ���ַ���) 
data = zeros(m,n);   %��ʼ�����ݾ���m*nά������Xij��ʾ���ʱ��е�j�������ڵ�i���������ֵĴ�����Ƶ����   
label = zeros(m,1);  %��ʼ�����ǩ���� 
for i = 1:m
    line = fgetl(fid);   %��ȡ��i����������
    nums = sscanf(line, '%d');  %�����ݷָ���������nums������
    label(i) = nums(1);    %���������ǩ�����ǩ����
    data(i,1 + cumsum(nums(2:2:end - 1))) = nums(3:2:end - 1);   %����Xij
end    
fclose(fid);   %�ر��ļ�
end
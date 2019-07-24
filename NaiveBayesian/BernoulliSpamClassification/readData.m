function [data,label] = readData(filename) 
%readData    读取数据

fid = fopen(filename);  %打开文件  
fgetl(fid);  %跳过数据文件第一行(标题) 

%读取数据集大小，数据集数量为m，数据的维度n(实际上n为单词表的单词总数)
mn = fscanf(fid, '%d %d\n', 2);    
m = mn(1);
n = mn(2);  

fgetl(fid);    %跳过数据文件第三行(单词表，实际上是一个很长的一个字符串) 
data = zeros(m,n);   %初始化数据矩阵，m*n维，其中Xij表示单词表中第j个单词在第i个样本出现的次数（频数）   
label = zeros(m,1);  %初始化类标签向量 
for i = 1:m
    line = fgetl(fid);   %读取第i个样本数据
    nums = sscanf(line, '%d');  %将数据分隔开，存入nums向量中
    label(i) = nums(1);    %样本分类标签存入标签向量
    data(i,1 + cumsum(nums(2:2:end - 1))) = nums(3:2:end - 1);   %计算Xij
end    
fclose(fid);   %关闭文件
end
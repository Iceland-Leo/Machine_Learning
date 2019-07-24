%读取数据
Alldata = textread('example.txt');
trainX = Alldata(1:150,1:2);
trainY = Alldata(1:150,3);
plot(trainX',trainY','*'); %画原数据散点图
hold on
testX = Alldata(151:200,1:2);
testY = Alldata(151:200,3);
theta = (trainX' * trainX) \ (trainX' * trainY); %theta = (X'*X)^-1 * X' * Y
predictY = trainX * theta;
plot(trainX',predictY');  %画回归函数图像
mse = norm(testX * theta - testY);
%��ȡ����
Alldata = textread('example.txt');
trainX = Alldata(1:150,1:2);
trainY = Alldata(1:150,3);
plot(trainX',trainY','*'); %��ԭ����ɢ��ͼ
hold on
testX = Alldata(151:200,1:2);
testY = Alldata(151:200,3);
theta = (trainX' * trainX) \ (trainX' * trainY); %theta = (X'*X)^-1 * X' * Y
predictY = trainX * theta;
plot(trainX',predictY');  %���ع麯��ͼ��
mse = norm(testX * theta - testY);
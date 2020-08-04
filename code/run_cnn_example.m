clear all; close all; clc;
addpath('./util');

% ��mnist����
load mnist_uint8;
train_x   =  double(reshape(train_x',28,28,60000))/255;
test_x    =  double(reshape(test_x',28,28,10000))/255;
train_y  =  double(train_y');
test_y   =  double(test_y');

% cnn�Ĳ���
cnn.layers = {  
    struct('type', 'i') %input layer  
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % convolution layer  
    struct('type', 's', 'scale', 2) % subsampling layer  
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % convolution layer  
    struct('type', 's', 'scale', 2) % subsampling layer  
};

% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y);

% ���糬����
opts.alpha           = 1   ; %ѧϰ�ʲ���
opts.batchsize     = 20; %ÿһ�����ݵ�����
opts.numepochs = 1  ; %������ѭ������

IterMax   = 2; % �������ݼ��ϵĵ�������
precision = zeros(1,IterMax);
disp('======start training======');

for iter = 1:IterMax
    
    
    % Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    % Ȼ����ò�������������
    [er, bad] = cnntest(cnn, test_x, test_y);
    
    disp(['% ',num2str(iter),' %', '---------%׼ȷ��',num2str(100 - er*100) '%---------']);
    precision(iter) = 1-er;
    
end

plot(precision);

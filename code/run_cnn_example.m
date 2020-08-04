clear all; close all; clc;
addpath('./util');

% 读mnist数据
load mnist_uint8;
train_x   =  double(reshape(train_x',28,28,60000))/255;
test_x    =  double(reshape(test_x',28,28,10000))/255;
train_y  =  double(train_y');
test_y   =  double(test_y');

% cnn的参数
cnn.layers = {  
    struct('type', 'i') %input layer  
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % convolution layer  
    struct('type', 's', 'scale', 2) % subsampling layer  
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % convolution layer  
    struct('type', 's', 'scale', 2) % subsampling layer  
};

% 这里把cnn的设置给cnnsetup，它会据此构建一个完整的CNN网络，并返回
cnn = cnnsetup(cnn, train_x, train_y);

% 网络超参数
opts.alpha           = 1   ; %学习率步长
opts.batchsize     = 20; %每一批数据的数量
opts.numepochs = 1  ; %迭代大循环次数

IterMax   = 2; % 整个数据集上的迭代次数
precision = zeros(1,IterMax);
disp('======start training======');

for iter = 1:IterMax
    
    
    % 然后开始把训练样本给它，开始训练这个CNN网络
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    % 然后就用测试样本来测试
    [er, bad] = cnntest(cnn, test_x, test_y);
    
    disp(['% ',num2str(iter),' %', '---------%准确率',num2str(100 - er*100) '%---------']);
    precision(iter) = 1-er;
    
end

plot(precision);

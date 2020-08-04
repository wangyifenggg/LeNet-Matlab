function net = net_forward(net,x)

%各变量含义
%student:wang yi feng
%ID:2019E8020261077
%x : （28*28*20）1batch的输入数据
%struct net : 
%5*1 layers:    
%   layer 1: type 'i' 类型为输入层
%            a (28*28*20) 输入1个batch的像素值，（长*宽*batchsize）
%   layer 2: type 'c' 类型为卷积层
%            a (24*24*20)*6 卷积一次得到的6个特征图，因为valid卷积边长减小4
%            b 1*6 卷积一次得到的6个特征图的偏置量
%            ad (24*24*20)*6 计算出特征图的梯度
%            db 1*6 计算出特征图的偏置量的梯度
%            db_old 1*6 上一次的特征图的偏置量的梯度
%            k 5*5*6 6个5*5的卷积核权重矩阵
%            dk_old 5*5*6 上一次的6个5*5的卷积核权重矩阵梯度
%            dk 5*5*6 6个5*5的卷积核权重矩阵梯度
%            kernelsize 5 卷积核大小为5
%            outputmaps 6 特征图的数目为6
%   layer 3: type 's' 类型为池化层
%            scale 2 池化尺度
%            a (12*12*20)*6 池化后的6个特征图，长宽缩小一倍
%            b 1*6 池化后的6个特征图的偏置量
%            d (12*12*20)*6 计算出特征图的梯度
%            outputmaps 6 特征图的数目为6
%   layer 4: type 'c' 类型为卷积层
%            a (8*8*20)*12 卷积一次得到的12个特征图，因为valid卷积边长减小4
%            b 1*12 卷积一次得到的12个特征图的偏置量
%            db_old 1*12 上一次的特征图的偏置量的梯度
%            ad (8*8*20)*12 计算出特征图的梯度
%            db 1*12 计算出特征图的偏置量的梯度
%            k 5*5*12 12个5*5的卷积核权重矩阵
%            dk_old 5*5*12 上一次的12个5*5的卷积核权重矩阵梯度
%            dk 5*5*12 12个5*5的卷积核权重矩阵梯度
%            kernelsize 5 卷积核大小为5
%            outputmaps 12 特征图的数目为12
%   layer 5: type 's' 类型为池化层
%            scale 2 池化尺度
%            a (4*4*20)*12 池化后的6个特征图，长宽缩小一倍
%            b 1*12 池化后的6个特征图的偏置量
%            d (4*4*20)*12 计算出特征图的梯度
%            outputmaps 12 特征图的数目为12
%ffW 10*192 全连接权重矩阵
%ffb 10*1 全连接权重矩阵的偏置
%rL 1*2 学习率
%fv 192*20 第五层输出的特征图向量化的容器4*4*12*20->192*20
%o 10*20 输出预测值，（10*batchsize）
%e 10*20 输出预测值与真实值的偏差，（10*batchsize）
%L 1*1 学习率
%do 10*20 输出预测值的梯度，（10*batchsize）
%fvd 192*20 第五层输出的特征图向量化的梯度
%dffW 10*192 全连接权重矩阵的梯度
%dffb 10*1 全连接权重矩阵的偏置的梯度
%dffW_old 10*192 上一次的全连接权重矩阵的梯度
%dffb_old 10*1 上一次的全连接权重矩阵的偏置的梯度
[m,~]=size(net.layers);
net.layers{1}.a{1}=x;
batchsize=size(x,3);
net.layers{1}.outputmaps=1;
input_features=1;

for i=2:m
    
    if net.layers{i}.type=='c'%如果是卷积层
        %fprintf('conv');
        output_features=net.layers{i}.outputmaps;%输出特征图数目
        for j=1:output_features
            next_length = size(net.layers{i - 1}.a{1},1) - net.layers{i}.kernelsize + 1;%下一个特征图的长宽
            next_a=zeros(next_length,next_length,batchsize);%下一个特征图的容器
            for k=1:input_features
                next_a=next_a+convn(net.layers{i - 1}.a{k},net.layers{i}.k{k}{j}, 'valid');%把每个输入特征图卷积后的结果叠加
            end
             net.layers{i}.a{j} = sigm(next_a + net.layers{i}.b{j});% 加入偏置与非线性激活
        end
        input_features = net.layers{i}.outputmaps;%下一层输入的maps等于上一层输出的maps
    end
    if net.layers{i}.type=='s'        
        %fprintf('sample');
        scale=net.layers{i}.scale;%池化尺度
        POOL=ones(scale,scale)/(scale*scale);%池化矩阵
        
        for n=1:input_features
            a=net.layers{i-1}.a{n};
            next_length=size(net.layers{i - 1}.a{1},1)/scale;%下一个特征图大小
            next_a=zeros(next_length,next_length,batchsize);%下一个特征图的容器        
            for n1=1:next_length
                for n2=1:next_length
                    block = a((n1-1)*scale+1:n1*scale,(n2-1)*scale+1:n2*scale,:) .* repmat(POOL,1,1,batchsize);%池化矩阵与对应特征图卷积
                    next_a(n1,n2,:) = sum(sum(block));  
                end
            
            end
            net.layers{i}.a{n}=next_a;
        end
        net.layers{i}.outputmaps=input_features;%池化层的特征图数目不变
    end
    
end
net.fv = [];
for j = 1 : net.layers{m-1}.outputmaps    %将特征图向量化
    net.fv = [net.fv; reshape(net.layers{m}.a{j}, [], batchsize)];
end
    
%通过全连接层计算输出
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));


end
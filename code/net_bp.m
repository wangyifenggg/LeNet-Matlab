function net = net_bp(net,y)
%student:wang yi feng
%ID:2019E8020261077
[m,~]=size(net.layers);
net.e=net.o-y;%误差
net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%将E归一化
net.do=net.e.*(net.o.*(1-net.o));%输出的梯度=误差*激活函数的导数
net.fvd = (net.ffW' * net.do); %全连接层的梯度=权重矩阵*输出的梯度
if net.layers{m}.type=='c'
    net.fvd = net.fvd .* (net.fv .* (1 - net.fv));%全连接层的梯度=全连接层的梯度*激活函数的导数
end
net.dffW_old=net.dffW;
net.dffb_old=net.dffb;
net.dffW = net.do * (net.fv)' / size(net.do, 2);
net.dffb = mean(net.do, 2);
[l,~,batchsize] = size(net.layers{m}.a{1});%得到最后特征图的边长和batchsize
for i=1:net.layers{m}.outputmaps
    net.layers{m}.ad{i}=reshape(net.fvd(((i - 1) * l * l + 1) : i * l * l, :),l,l,batchsize);
    
end

for n=(m-1):-1:1
    if net.layers{n}.type=='c'
        for j = 1 : net.layers{n}.outputmaps%进行上采样运算，此处与前传部分反过来了
            pre_ad=expand(net.layers{n + 1}.ad{j}, [net.layers{n + 1}.scale net.layers{n + 1}.scale 1]) / net.layers{n + 1}.scale ^ 2;%将上一层特征图的每个点均分成scale ^ 2个
            net.layers{n}.ad{j} = net.layers{n}.a{j} .* (1 - net.layers{n}.a{j}) .* (pre_ad);%传到下一层特征图的梯度
        end
    elseif net.layers{n}.type=='s'
        for i=1:net.layers{n}.outputmaps
            pre_ad = zeros(size(net.layers{n}.a{1}));
            for j=1:net.layers{n+1}.outputmaps
                pre_ad=pre_ad+convn(net.layers{n + 1}.ad{j}, rot90(net.layers{n + 1}.k{i}{j},2), 'full');%卷积核翻转180度与上一层特征图的梯度进行卷积
            end
            net.layers{n}.ad{i} = pre_ad;
        end
    end
end

for n=2:m
    if net.layers{n}.type=='c'
        for i=1:net.layers{n-1}.outputmaps
           for j=1: net.layers{n}.outputmaps
               net.layers{n}.dk_old{i}{j} = net.layers{n}.dk{i}{j};%保存上一次的卷积核梯度
               net.layers{n}.dk{i}{j} = convn(flipall(net.layers{n - 1}.a{i}), net.layers{n}.ad{j}, 'valid') / size(net.layers{n}.ad{j}, 3);%计算本次的卷积核梯度
           end
        end
        for j=1: net.layers{n}.outputmaps
            net.layers{n}.db_old{j}=net.layers{n}.db{j};%保存上一次的偏置梯度
            net.layers{n}.db{j} = sum(net.layers{n}.ad{j}(:)) / size(net.layers{n}.ad{j}, 3);%计算本次的偏置梯度
        end
    end
end

end


%以SGD与BP算法训练卷积神经网络
function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    
    for i = 1 : opts.numepochs
        tic;
        kk = randperm(m);
        
        %对于所有的batch
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            net=net_forward(net, batch_x);%前向传播函数
            net=net_bp(net,batch_y);%反向传播函数
            net=net_gradupdate(net,opts);%使用Momentum法更新神经网络权值

            %三个主函数
            %net = cnnff(net, batch_x);            %前向传播函数
            %net = cnnbp(net, batch_y);           %反向传播函数
            %net = cnnapplygrads(net, opts);  %根据梯度更新神经网络权值
            
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end

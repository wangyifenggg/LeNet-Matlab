%根据梯度更新神经网络参数
function net = cnnapplygrads(net, opts)

    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')%对于卷积层
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j}; %更新w
                end
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};%更新b
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;%全连接层w
    net.ffb  = net.ffb - opts.alpha * net.dffb    ;%全连接层b
    
    
end

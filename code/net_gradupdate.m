function net = net_gradupdate(net,opts)
%student:wang yi feng
%ID:2019E8020261077    
    [m,~]=size(net.layers);
    beta=0.9;%动量项的参数，beta=0时为SGD,0<beta<1时为Momentum
    for n = 2 : m
        if net.layers{n}.type== 'c'%对于卷积层
            for j = 1 : net.layers{n}.outputmaps
                for i = 1 : net.layers{n - 1}.outputmaps
                    net.layers{n}.k{i}{j} = net.layers{n}.k{i}{j} - opts.alpha * net.layers{n}.dk{i}{j} - beta*net.layers{n}.dk_old{i}{j} ; %更新w
                end
                net.layers{n}.b{j} = net.layers{n}.b{j} - opts.alpha * net.layers{n}.db{j}-beta*net.layers{n}.db_old{j};%更新b
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW - beta*net.dffW_old;%全连接层w
    net.ffb  = net.ffb - opts.alpha * net.dffb - beta*net.dffb_old;%全连接层b


end


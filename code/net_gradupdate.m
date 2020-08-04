function net = net_gradupdate(net,opts)
%student:wang yi feng
%ID:2019E8020261077    
    [m,~]=size(net.layers);
    beta=0.9;%������Ĳ�����beta=0ʱΪSGD,0<beta<1ʱΪMomentum
    for n = 2 : m
        if net.layers{n}.type== 'c'%���ھ����
            for j = 1 : net.layers{n}.outputmaps
                for i = 1 : net.layers{n - 1}.outputmaps
                    net.layers{n}.k{i}{j} = net.layers{n}.k{i}{j} - opts.alpha * net.layers{n}.dk{i}{j} - beta*net.layers{n}.dk_old{i}{j} ; %����w
                end
                net.layers{n}.b{j} = net.layers{n}.b{j} - opts.alpha * net.layers{n}.db{j}-beta*net.layers{n}.db_old{j};%����b
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW - beta*net.dffW_old;%ȫ���Ӳ�w
    net.ffb  = net.ffb - opts.alpha * net.dffb - beta*net.dffb_old;%ȫ���Ӳ�b


end


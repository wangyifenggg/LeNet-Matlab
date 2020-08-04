%�����ݶȸ������������
function net = cnnapplygrads(net, opts)

    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')%���ھ����
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j}; %����w
                end
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};%����b
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;%ȫ���Ӳ�w
    net.ffb  = net.ffb - opts.alpha * net.dffb    ;%ȫ���Ӳ�b
    
    
end

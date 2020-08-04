%��SGD��BP�㷨ѵ�����������
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
        
        %�������е�batch
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            net=net_forward(net, batch_x);%ǰ�򴫲�����
            net=net_bp(net,batch_y);%���򴫲�����
            net=net_gradupdate(net,opts);%ʹ��Momentum������������Ȩֵ

            %����������
            %net = cnnff(net, batch_x);            %ǰ�򴫲�����
            %net = cnnbp(net, batch_y);           %���򴫲�����
            %net = cnnapplygrads(net, opts);  %�����ݶȸ���������Ȩֵ
            
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end

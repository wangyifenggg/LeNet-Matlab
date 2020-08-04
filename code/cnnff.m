function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  对于每一个层
        
        if strcmp(net.layers{l}.type, 'c')%如果是卷积层
            
            for j = 1 : net.layers{l}.outputmaps   %  对于每一个输出map                
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);% 创建一个暂时map
                
                for i = 1 : inputmaps   % 对于每一个输入map                 
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');   %  与卷积核卷积加到输出map
                end
                
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});% 加入偏置与非线性激活
            end
            
            inputmaps = net.layers{l}.outputmaps;%下一层输入的maps等于上一层输出的maps
    
        elseif strcmp(net.layers{l}.type, 's')%如果采样层
            
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   % 通过卷积实现采样
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    % 把卷积特征向量化
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    %通过全连接层计算输出
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

end

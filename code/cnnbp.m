%反向传播函数
function net = cnnbp(net, y)
    n = numel(net.layers);

    net.e = net.o - y;%计算输出错误E  
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%将E归一化

    net.od = net.e .* (net.o .* (1 - net.o));   %计算输出的 delta
    net.fvd = (net.ffW' * net.od);                 %计算W'*delta
    if strcmp(net.layers{n}.type, 'c')             %卷积层乘上激活函数
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %把deltas reshape成输出图一样格式
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1%对于所有层
        if strcmp(net.layers{l}.type, 'c')%如果是卷积层
            
            for j = 1 : numel(net.layers{l}.a)%进行上采样运算，此处与前传部分反过来了
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
            
        elseif strcmp(net.layers{l}.type, 's')%如果是采样层
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot90(net.layers{l + 1}.k{i}{j},2), 'full');%通过卷积反传delta
                     %z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');%通过卷积反传delta
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %根据delta计算梯度，此处变量是dk, db
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk_old{i}{j} = net.layers{l}.dk{i}{j};
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end

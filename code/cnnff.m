function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  ����ÿһ����
        
        if strcmp(net.layers{l}.type, 'c')%����Ǿ����
            
            for j = 1 : net.layers{l}.outputmaps   %  ����ÿһ�����map                
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);% ����һ����ʱmap
                
                for i = 1 : inputmaps   % ����ÿһ������map                 
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');   %  �����˾���ӵ����map
                end
                
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});% ����ƫ��������Լ���
            end
            
            inputmaps = net.layers{l}.outputmaps;%��һ�������maps������һ�������maps
    
        elseif strcmp(net.layers{l}.type, 's')%���������
            
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   % ͨ�����ʵ�ֲ���
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    % �Ѿ������������
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    %ͨ��ȫ���Ӳ�������
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

end

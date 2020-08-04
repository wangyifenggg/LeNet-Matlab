
function [er, bad] = cnntest(net, x, y)
    
    %前传
    net = cnnff(net, x);
    
    [~, h] = max(net.o);%对于输出向量取最大标号
    [~, a] = max(y);
    bad = find(h ~= a); %找出与标号不一致的输出

    er = numel(bad) / size(y, 2);%计算错误率
end

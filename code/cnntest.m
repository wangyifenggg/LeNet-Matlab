
function [er, bad] = cnntest(net, x, y)
    
    %ǰ��
    net = cnnff(net, x);
    
    [~, h] = max(net.o);%�����������ȡ�����
    [~, a] = max(y);
    bad = find(h ~= a); %�ҳ����Ų�һ�µ����

    er = numel(bad) / size(y, 2);%���������
end

function net = net_bp(net,y)
%student:wang yi feng
%ID:2019E8020261077
[m,~]=size(net.layers);
net.e=net.o-y;%���
net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%��E��һ��
net.do=net.e.*(net.o.*(1-net.o));%������ݶ�=���*������ĵ���
net.fvd = (net.ffW' * net.do); %ȫ���Ӳ���ݶ�=Ȩ�ؾ���*������ݶ�
if net.layers{m}.type=='c'
    net.fvd = net.fvd .* (net.fv .* (1 - net.fv));%ȫ���Ӳ���ݶ�=ȫ���Ӳ���ݶ�*������ĵ���
end
net.dffW_old=net.dffW;
net.dffb_old=net.dffb;
net.dffW = net.do * (net.fv)' / size(net.do, 2);
net.dffb = mean(net.do, 2);
[l,~,batchsize] = size(net.layers{m}.a{1});%�õ��������ͼ�ı߳���batchsize
for i=1:net.layers{m}.outputmaps
    net.layers{m}.ad{i}=reshape(net.fvd(((i - 1) * l * l + 1) : i * l * l, :),l,l,batchsize);
    
end

for n=(m-1):-1:1
    if net.layers{n}.type=='c'
        for j = 1 : net.layers{n}.outputmaps%�����ϲ������㣬�˴���ǰ�����ַ�������
            pre_ad=expand(net.layers{n + 1}.ad{j}, [net.layers{n + 1}.scale net.layers{n + 1}.scale 1]) / net.layers{n + 1}.scale ^ 2;%����һ������ͼ��ÿ������ֳ�scale ^ 2��
            net.layers{n}.ad{j} = net.layers{n}.a{j} .* (1 - net.layers{n}.a{j}) .* (pre_ad);%������һ������ͼ���ݶ�
        end
    elseif net.layers{n}.type=='s'
        for i=1:net.layers{n}.outputmaps
            pre_ad = zeros(size(net.layers{n}.a{1}));
            for j=1:net.layers{n+1}.outputmaps
                pre_ad=pre_ad+convn(net.layers{n + 1}.ad{j}, rot90(net.layers{n + 1}.k{i}{j},2), 'full');%����˷�ת180������һ������ͼ���ݶȽ��о��
            end
            net.layers{n}.ad{i} = pre_ad;
        end
    end
end

for n=2:m
    if net.layers{n}.type=='c'
        for i=1:net.layers{n-1}.outputmaps
           for j=1: net.layers{n}.outputmaps
               net.layers{n}.dk_old{i}{j} = net.layers{n}.dk{i}{j};%������һ�εľ�����ݶ�
               net.layers{n}.dk{i}{j} = convn(flipall(net.layers{n - 1}.a{i}), net.layers{n}.ad{j}, 'valid') / size(net.layers{n}.ad{j}, 3);%���㱾�εľ�����ݶ�
           end
        end
        for j=1: net.layers{n}.outputmaps
            net.layers{n}.db_old{j}=net.layers{n}.db{j};%������һ�ε�ƫ���ݶ�
            net.layers{n}.db{j} = sum(net.layers{n}.ad{j}(:)) / size(net.layers{n}.ad{j}, 3);%���㱾�ε�ƫ���ݶ�
        end
    end
end

end


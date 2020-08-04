function net = net_forward(net,x)

%����������
%student:wang yi feng
%ID:2019E8020261077
%x : ��28*28*20��1batch����������
%struct net : 
%5*1 layers:    
%   layer 1: type 'i' ����Ϊ�����
%            a (28*28*20) ����1��batch������ֵ������*��*batchsize��
%   layer 2: type 'c' ����Ϊ�����
%            a (24*24*20)*6 ���һ�εõ���6������ͼ����Ϊvalid����߳���С4
%            b 1*6 ���һ�εõ���6������ͼ��ƫ����
%            ad (24*24*20)*6 ���������ͼ���ݶ�
%            db 1*6 ���������ͼ��ƫ�������ݶ�
%            db_old 1*6 ��һ�ε�����ͼ��ƫ�������ݶ�
%            k 5*5*6 6��5*5�ľ����Ȩ�ؾ���
%            dk_old 5*5*6 ��һ�ε�6��5*5�ľ����Ȩ�ؾ����ݶ�
%            dk 5*5*6 6��5*5�ľ����Ȩ�ؾ����ݶ�
%            kernelsize 5 ����˴�СΪ5
%            outputmaps 6 ����ͼ����ĿΪ6
%   layer 3: type 's' ����Ϊ�ػ���
%            scale 2 �ػ��߶�
%            a (12*12*20)*6 �ػ����6������ͼ��������Сһ��
%            b 1*6 �ػ����6������ͼ��ƫ����
%            d (12*12*20)*6 ���������ͼ���ݶ�
%            outputmaps 6 ����ͼ����ĿΪ6
%   layer 4: type 'c' ����Ϊ�����
%            a (8*8*20)*12 ���һ�εõ���12������ͼ����Ϊvalid����߳���С4
%            b 1*12 ���һ�εõ���12������ͼ��ƫ����
%            db_old 1*12 ��һ�ε�����ͼ��ƫ�������ݶ�
%            ad (8*8*20)*12 ���������ͼ���ݶ�
%            db 1*12 ���������ͼ��ƫ�������ݶ�
%            k 5*5*12 12��5*5�ľ����Ȩ�ؾ���
%            dk_old 5*5*12 ��һ�ε�12��5*5�ľ����Ȩ�ؾ����ݶ�
%            dk 5*5*12 12��5*5�ľ����Ȩ�ؾ����ݶ�
%            kernelsize 5 ����˴�СΪ5
%            outputmaps 12 ����ͼ����ĿΪ12
%   layer 5: type 's' ����Ϊ�ػ���
%            scale 2 �ػ��߶�
%            a (4*4*20)*12 �ػ����6������ͼ��������Сһ��
%            b 1*12 �ػ����6������ͼ��ƫ����
%            d (4*4*20)*12 ���������ͼ���ݶ�
%            outputmaps 12 ����ͼ����ĿΪ12
%ffW 10*192 ȫ����Ȩ�ؾ���
%ffb 10*1 ȫ����Ȩ�ؾ����ƫ��
%rL 1*2 ѧϰ��
%fv 192*20 ��������������ͼ������������4*4*12*20->192*20
%o 10*20 ���Ԥ��ֵ����10*batchsize��
%e 10*20 ���Ԥ��ֵ����ʵֵ��ƫ���10*batchsize��
%L 1*1 ѧϰ��
%do 10*20 ���Ԥ��ֵ���ݶȣ���10*batchsize��
%fvd 192*20 ��������������ͼ���������ݶ�
%dffW 10*192 ȫ����Ȩ�ؾ�����ݶ�
%dffb 10*1 ȫ����Ȩ�ؾ����ƫ�õ��ݶ�
%dffW_old 10*192 ��һ�ε�ȫ����Ȩ�ؾ�����ݶ�
%dffb_old 10*1 ��һ�ε�ȫ����Ȩ�ؾ����ƫ�õ��ݶ�
[m,~]=size(net.layers);
net.layers{1}.a{1}=x;
batchsize=size(x,3);
net.layers{1}.outputmaps=1;
input_features=1;

for i=2:m
    
    if net.layers{i}.type=='c'%����Ǿ����
        %fprintf('conv');
        output_features=net.layers{i}.outputmaps;%�������ͼ��Ŀ
        for j=1:output_features
            next_length = size(net.layers{i - 1}.a{1},1) - net.layers{i}.kernelsize + 1;%��һ������ͼ�ĳ���
            next_a=zeros(next_length,next_length,batchsize);%��һ������ͼ������
            for k=1:input_features
                next_a=next_a+convn(net.layers{i - 1}.a{k},net.layers{i}.k{k}{j}, 'valid');%��ÿ����������ͼ�����Ľ������
            end
             net.layers{i}.a{j} = sigm(next_a + net.layers{i}.b{j});% ����ƫ��������Լ���
        end
        input_features = net.layers{i}.outputmaps;%��һ�������maps������һ�������maps
    end
    if net.layers{i}.type=='s'        
        %fprintf('sample');
        scale=net.layers{i}.scale;%�ػ��߶�
        POOL=ones(scale,scale)/(scale*scale);%�ػ�����
        
        for n=1:input_features
            a=net.layers{i-1}.a{n};
            next_length=size(net.layers{i - 1}.a{1},1)/scale;%��һ������ͼ��С
            next_a=zeros(next_length,next_length,batchsize);%��һ������ͼ������        
            for n1=1:next_length
                for n2=1:next_length
                    block = a((n1-1)*scale+1:n1*scale,(n2-1)*scale+1:n2*scale,:) .* repmat(POOL,1,1,batchsize);%�ػ��������Ӧ����ͼ���
                    next_a(n1,n2,:) = sum(sum(block));  
                end
            
            end
            net.layers{i}.a{n}=next_a;
        end
        net.layers{i}.outputmaps=input_features;%�ػ��������ͼ��Ŀ����
    end
    
end
net.fv = [];
for j = 1 : net.layers{m-1}.outputmaps    %������ͼ������
    net.fv = [net.fv; reshape(net.layers{m}.a{j}, [], batchsize)];
end
    
%ͨ��ȫ���Ӳ�������
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));


end
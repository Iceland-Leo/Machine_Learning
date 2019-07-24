function [j] = selectJ( i,Ei,trainlabel,nonBoundList,kernel,model )
%selectJ ѡ��ڶ���Ҫ�ı��alpha���±�ֵ

j = -1;
max = 0;

%���ȸ�������ʽ����ѡ��߽��alpha���������ѡ��һ����
if length(nonBoundList) > 1
    for k = 1:length(nonBoundList)
        if nonBoundList(k) == i
            continue;
        end
        E = Error(nonBoundList(k),trainlabel,kernel,model);
        deltaE = abs(Ei - E);
        if deltaE > max
            max = deltaE;
            j = nonBoundList(k);
        end
    end
    return;
else
    while 1
        j = unidrnd(length(model.alpha));
        if j ~= i
            break;
        end
    end
    return;
end

end


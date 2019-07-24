function [j] = selectJ( i,Ei,trainlabel,nonBoundList,kernel,model )
%selectJ 选择第二个要改变的alpha的下标值

j = -1;
max = 0;

%首先根据启发式规则选择边界的alpha，否则随机选择一个点
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


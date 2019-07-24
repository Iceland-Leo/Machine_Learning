function [changed,model,nonBoundList] = innerLoop(i,trainlabel,oldnonBoundList,kernel,C,epsilon,oldmodel)
%innerLoop SMO的内层循环

Ei = Error(i,trainlabel,kernel,oldmodel); %计算误差
r = trainlabel(i) * Ei;  %计算r值
%如果不满足KKT条件,则更新,若满足,则不更新
if ((r > epsilon && oldmodel.alpha(i) > 0) || (r < -epsilon && oldmodel.alpha(i) < C))
    j = selectJ(i,Ei,trainlabel,oldnonBoundList,kernel,oldmodel);  %第二个要改变的alpha的下标值
    Ej = Error(j,trainlabel,kernel,oldmodel);
    oldAlphaI = oldmodel.alpha(i);
    oldAlphaJ = oldmodel.alpha(j);
    %分情况讨论alpha改变值的上下界
    if((trainlabel(i) ~= trainlabel(j)))
		Low = max(0, oldAlphaJ - oldAlphaI);
		High = min(C, C + oldAlphaJ - oldAlphaI);
    else
		Low = max(0, oldAlphaJ + oldAlphaI - C);
		High = min(C, oldAlphaJ + oldAlphaI);
    end
    
    eta = kernel(i,i) + kernel(j,j) - 2 * kernel(i,j);
    %如果eta小于等于0,则不需要改变alpha值
    if eta <= 0
        changed = 0;
        model = oldmodel;
        nonBoundList = oldnonBoundList;
        return;
    end
    
    oldmodel.alpha(j) = oldAlphaJ + (trainlabel(j) * (Ei - Ej) / eta);
    if oldmodel.alpha(j) > High
        oldmodel.alpha(j) = High;
    elseif oldmodel.alpha(j) < Low
        oldmodel.alpha(j) = Low;
    end
    %值未改变，则返回0
    if abs(oldmodel.alpha(j) - oldAlphaJ) < 1e-15
        changed = 0;
        model = oldmodel;
        nonBoundList = oldnonBoundList;
        return;
    end
    %存入边界alpha值向量中
    if oldmodel.alpha(j) > 0 && oldmodel.alpha(j) < C
        if ismember(j,oldnonBoundList) == 0
            oldnonBoundList = [oldnonBoundList;j];
        end
    end
    
    oldmodel.alpha(i) = oldAlphaI + trainlabel(i) * trainlabel(j) * (oldAlphaJ - oldmodel.alpha(j));
    %存入边界alpha值向量中
    if oldmodel.alpha(i) > 0 && oldmodel.alpha(i) < C
       if ismember(i,oldnonBoundList) == 0
           oldnonBoundList = [oldnonBoundList;i];
       end
    end
    %计算新偏置b
    bi = -Ei - trainlabel(i) * kernel(i,i) * (oldmodel.alpha(i) - oldAlphaI) - trainlabel(j) * kernel(j,i) * (oldmodel.alpha(j) - oldAlphaJ) + oldmodel.b;
	bj = -Ej - trainlabel(i) * kernel(i,j) * (oldmodel.alpha(i) - oldAlphaI) - trainlabel(j) * kernel(j,j) * (oldmodel.alpha(j) - oldAlphaJ) + oldmodel.b;
    if oldmodel.alpha(i) > 0 && oldmodel.alpha(i) < C
        oldmodel.b = bi;
    elseif oldmodel.alpha(j) > 0 && oldmodel.alpha(j) < C
        oldmodel.b = bj;
    else
        oldmodel.b = (bi + bj) / 2;
    end 
    changed = 1;
    model = oldmodel;
    nonBoundList = oldnonBoundList;
    return; 
else
    changed = 0;
    model = oldmodel;
    nonBoundList = oldnonBoundList;
    return;
end

end


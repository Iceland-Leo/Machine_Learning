function [changed,model,nonBoundList] = innerLoop(i,trainlabel,oldnonBoundList,kernel,C,epsilon,oldmodel)
%innerLoop SMO���ڲ�ѭ��

Ei = Error(i,trainlabel,kernel,oldmodel); %�������
r = trainlabel(i) * Ei;  %����rֵ
%���������KKT����,�����,������,�򲻸���
if ((r > epsilon && oldmodel.alpha(i) > 0) || (r < -epsilon && oldmodel.alpha(i) < C))
    j = selectJ(i,Ei,trainlabel,oldnonBoundList,kernel,oldmodel);  %�ڶ���Ҫ�ı��alpha���±�ֵ
    Ej = Error(j,trainlabel,kernel,oldmodel);
    oldAlphaI = oldmodel.alpha(i);
    oldAlphaJ = oldmodel.alpha(j);
    %���������alpha�ı�ֵ�����½�
    if((trainlabel(i) ~= trainlabel(j)))
		Low = max(0, oldAlphaJ - oldAlphaI);
		High = min(C, C + oldAlphaJ - oldAlphaI);
    else
		Low = max(0, oldAlphaJ + oldAlphaI - C);
		High = min(C, oldAlphaJ + oldAlphaI);
    end
    
    eta = kernel(i,i) + kernel(j,j) - 2 * kernel(i,j);
    %���etaС�ڵ���0,����Ҫ�ı�alphaֵ
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
    %ֵδ�ı䣬�򷵻�0
    if abs(oldmodel.alpha(j) - oldAlphaJ) < 1e-15
        changed = 0;
        model = oldmodel;
        nonBoundList = oldnonBoundList;
        return;
    end
    %����߽�alphaֵ������
    if oldmodel.alpha(j) > 0 && oldmodel.alpha(j) < C
        if ismember(j,oldnonBoundList) == 0
            oldnonBoundList = [oldnonBoundList;j];
        end
    end
    
    oldmodel.alpha(i) = oldAlphaI + trainlabel(i) * trainlabel(j) * (oldAlphaJ - oldmodel.alpha(j));
    %����߽�alphaֵ������
    if oldmodel.alpha(i) > 0 && oldmodel.alpha(i) < C
       if ismember(i,oldnonBoundList) == 0
           oldnonBoundList = [oldnonBoundList;i];
       end
    end
    %������ƫ��b
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


function [E] = Error(i,trainlabel,kernel,model)
%Error �������
E = sum(model.alpha .* trainlabel .* kernel(:,i)) - trainlabel(i);
end


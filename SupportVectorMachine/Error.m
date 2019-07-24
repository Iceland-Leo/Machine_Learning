function [E] = Error(i,trainlabel,kernel,model)
%Error º∆À„ŒÛ≤Ó
E = sum(model.alpha .* trainlabel .* kernel(:,i)) - trainlabel(i);
end


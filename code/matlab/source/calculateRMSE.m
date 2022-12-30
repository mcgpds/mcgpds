function [e E] = calculateRMSE(X1,X2,type)

if(nargin<2)
    error('Too few arguments');
end
if(size(X1)~=size(X2))
    error('Dimensions mismatch');
end

e = sqrt(mean(mean((X1-X2).^2)));

end


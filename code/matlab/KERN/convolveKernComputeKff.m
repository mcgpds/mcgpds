function Kff = convolveKernComputeKff(convolveKern, X, X1)




%	Copyright (c) 2014 ZhaoJing
% 	last update 2014-03-16
D = convolveKern.outputDimension;
normfactor=1;
argExp=0;
if nargin<3
    for q=1:size(X,2)
        sigma_q=convolveKern.Lambda_k(q)+repmat(convolveKern.P_d(:,q),[1 D])+repmat(convolveKern.P_d(:,q)',[D 1]);% L+P+P'
        normfactor = normfactor.*sigma_q;
    end
    Kff = sqrt(prod(convolveKern.Lambda_k))./normfactor.^0.5.*(convolveKern.S*convolveKern.S');%|L|^0.5/|P+L|^0.5*S
else

for q=1:size(X,2)
        sigma_q=convolveKern.Lambda_k(q)+repmat(convolveKern.P_d(:,q),[1 D])+repmat(convolveKern.P_d(:,q)',[D 1]);% L+P+P'
        normfactor = normfactor.*sigma_q;
        X_q = X(:,q); 
        X1_q = X1(:,q)';
        distan = (X_q - X1_q).^2;
        argExp = argExp + sigma_q.^-1.*repmat(distan,[D,D]);
end
normfactor = sqrt(prod(convolveKern.Lambda_k))./normfactor.^0.5.*(convolveKern.S*convolveKern.S');%|L|^0.5/|P+L|^0.5*S
Kff = normfactor.*exp(-0.5*argExp); 
end
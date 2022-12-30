function Kfu = convolveKernComputeKfu(convolveKern, X, Z)




%	Copyright (c) 2014 ZhaoJing
% 	last update 2014-03-16
M = size(Z,1); 
D = convolveKern.outputDimension;
normfactor=1;
argExp=0;
for q=1:size(X,2)
        sigma_q=convolveKern.Lambda_k(q)+convolveKern.P_d(:,q);% L+P
        normfactor = normfactor.*sigma_q;
        X_q = X(:,q); 
        Z_q = Z(:,q)';
        distan = (repmat(X_q,[1 M]) - Z_q).^2;
        argExp = argExp + repmat(sigma_q.^-1, [1 M]).*repmat(distan,[D,1]);
end
normfactor = sqrt(prod(convolveKern.Lambda_k))./normfactor.^0.5.*convolveKern.S;%|L|^0.5/|P+L|^0.5*S
Kfu = repmat(normfactor,[1 M]).*exp(-0.5*argExp); 

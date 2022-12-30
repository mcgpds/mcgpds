function [K, Knovar, argExp] = rbfard2VardistPsi1Compute(rbfardKern, vardist, Z)

% RBFARD2VARDISTPSI1COMPUTE description.
%
%	Description:
%	[K, Knovar, argExp] = rbfard2VardistPsi1Compute(rbfardKern, vardist, Z)
%% 	rbfard2VardistPsi1Compute.m SVN version 583
% 	last update 2011-07-04T19:08:50.905190Z

% variational means
N  = size(vardist.means,1);
%  inducing variables 
M = size(Z,1); 

A = rbfardKern.inputScales;
         
argExp = zeros(N,M); 
normfactor = ones(N,1);
for q=1:vardist.latentDimension
%
    S_q = vardist.covars(:,q);  
    normfactor = normfactor.*(A(q)*S_q + 1);
    Mu_q = vardist.means(:,q); 
    Z_q = Z(:,q)';
    distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])).^2;
    argExp = argExp + repmat(A(q)./(A(q)*S_q + 1), [1 M]).*distan;
%
end
normfactor = normfactor.^0.5;
Knovar = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 
K = rbfardKern.variance*Knovar; 



function k0 = rbfard2VardistPsi0Compute(rbfardKern, vardist)

% RBFARD2VARDISTPSI0COMPUTE description.
%
%	Description:
%	k0 = rbfard2VardistPsi0Compute(rbfardKern, vardist)
%% 	rbfard2VardistPsi0Compute.m SVN version 583
% 	last update 2011-07-04T19:08:50.834573Z
  
% variational means

k0 = vardist.numData*rbfardKern.variance; 



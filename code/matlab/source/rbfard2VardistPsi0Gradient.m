function [gKern, gVarmeans, gVarcovars] = rbfard2VardistPsi0Gradient(rbfard2Kern, vardist, covGrad, rbfard2VardistPsi2Gradient)

% RBFARD2VARDISTPSI0GRADIENT Description
%
%	Description:
%	[gKern, gVarmeans, gVarcovars] = rbfard2VardistPsi0Gradient(rbfard2Kern, vardist, covGrad)
%% 	rbfard2VardistPsi0Gradient.m SVN version 583
% 	last update 2011-07-04T19:08:50.874563Z
gKern = zeros(1,rbfard2Kern.nParams); 
gKern(1) = covGrad*vardist.numData;
 
gVarmeans = zeros(1,prod(size(vardist.means))); 
gVarcovars = zeros(1,prod(size(vardist.means))); 





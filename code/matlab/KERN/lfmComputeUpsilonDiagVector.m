function upsilon = lfmComputeUpsilonDiagVector(gamma, sigma2, t)

% LFMCOMPUTEUPSILONDIAGVECTOR
%
%	Description:
%	


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	lfmComputeUpsilonDiagVector.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z

upsilon = exp(-gamma*t).*lfmComputeUpsilonVector(-gamma ,sigma2, t);
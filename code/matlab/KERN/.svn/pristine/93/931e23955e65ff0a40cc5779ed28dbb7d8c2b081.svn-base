function dUpsilonS = lfmapGradientSigmaUpsilonVector(gamma, sigma2, t)

% LFMAPGRADIENTSIGMAUPSILONVECTOR Gradient of upsilon vector ap wrt sigma
%
%	Description:
%
%	UPSILON = LFMAPGRADIENTSIGMAUPSILONVECTOR(GAMMA, SIGMA2, T) computes
%	the gradient of a portion of the LFMAP kernel.
%	 Returns:
%	  UPSILON - result of this subcomponent of the kernel for the given
%	   values.
%	 Arguments:
%	  GAMMA - Gamma value for system.
%	  SIGMA2 - length scale of latent process.
%	  T - first time input (number of time points x 1).
%	
%
%	See also
%	LFMAPCOMPUTEUPSILONMATRIX.M


%	Copyright (c) 2010 Mauricio Alvarez
% 	lfmapGradientSigmaUpsilonVector.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z


dUpsilon = lfmGradientSigmaUpsilonVector(gamma, sigma2, t);

dUpsilonS = (gamma^2)*dUpsilon + (2/(sqrt(pi)*sigma2))* ...
    exp(-(t.^2)./sigma2).*(gamma + 2*t/sigma2).*(1-2*(t.^2)/sigma2) ...
    + (8/(sqrt(pi)*sigma2^2))*t.*exp(-(t.^2)./sigma2);

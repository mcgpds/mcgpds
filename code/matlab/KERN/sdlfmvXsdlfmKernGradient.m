function [g1,g2, covGradLocal] = sdlfmvXsdlfmKernGradient(sdlfmvKern1, sdlfmKern2, ...
    t1, t2, covGrad, covIC)

% SDLFMVXSDLFMKERNGRADIENT Gradients of cross kernel between 2 SDLFM kernels.
%
%	Description:
%
%	[G1, G2, COVGRADLOCAL] = SDLFMVXSDLFMKERNGRADIENT(SDLFMVKERN1,
%	SDLFMKERN2, T1, COVGRAD, COVIC) computes a cross gradient for a
%	cross kernel between two switching dynamical LFM kernels for the
%	multiple output kernel. The first SDLFM corresponds to the velocity
%	and the second to a position.
%	 Returns:
%	  G1 - gradient of the parameters of the first kernel, for ordering
%	   see lfmKernExtractParam.
%	  G2 - gradient of the parameters of the second kernel, for ordering
%	   see lfmKernExtractParam.
%	  COVGRADLOCAL - partial covariance wrt the first initial conditions
%	 Arguments:
%	  SDLFMVKERN1 - the kernel structure associated with the first SDLFM
%	   kernel (velocity).
%	  SDLFMKERN2 - the kernel structure associated with the second SDLFM
%	   kernel (position).
%	  T1 - inputs for which kernel is to be computed.
%	  COVGRAD - gradient of the objective function with respect to the
%	   elements of the cross kernel matrix.
%	  COVIC - covariance for the initial conditions
%
%	[G1, G2, COVGRADLOCAL] = SDLFMVXSDLFMKERNGRADIENT(SDLFMVKERN1,
%	SDLFMKERN2, T1, T2, COVGRAD, COVIC) computes a cross gradient for a
%	cross kernel between two switching dynamical LFM kernels for the
%	multiple output kernel. The first SDLFM corresponds to the velocity
%	and the second to a position.
%	 Returns:
%	  G1 - gradient of the parameters of the first kernel, for ordering
%	   see lfmKernExtractParam.
%	  G2 - gradient of the parameters of the second kernel, for ordering
%	   see lfmKernExtractParam.
%	  COVGRADLOCAL - partial covariance wrt the first initial conditions
%	 Arguments:
%	  SDLFMVKERN1 - the kernel structure associated with the first SDLFM
%	   kernel (velocity).
%	  SDLFMKERN2 - the kernel structure associated with the second SDLFM
%	   kernel (position).
%	  T1 - row inputs for which kernel is to be computed.
%	  T2 - column inputs for which kernel is to be computed.
%	  COVGRAD - gradient of the objective function with respect to the
%	   elements of the cross kernel matrix.
%	  COVIC - covariance for the initial conditions


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	sdlfmvXsdlfmKernGradient.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z

if nargin == 4
    covGrad = t2;
    t2 = t1;
end

[g1, g2, covGradLocal] = sdlfmXsdlfmKernGradient(sdlfmvKern1, sdlfmKern2, t1, ...
    t2, covGrad, covIC, 'VelPos');
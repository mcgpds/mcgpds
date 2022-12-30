function dUpsilon = lfmaaGradientUpsilonMatrix(gamma, sigma2, t1, ...
    t2, mode, upsilon)

% LFMAAGRADIENTUPSILONMATRIX Gradient upsilon matrix accel. accel.
%
%	Description:
%
%	UPSILON = LFMAAGRADIENTUPSILONMATRIX(GAMMA, SIGMA2, T1, T2, UPSILON,
%	MODE) computes the gradient of a portion of the LFMAA kernel.
%	 Returns:
%	  UPSILON - result of this subcomponent of the kernel for the given
%	   values.
%	 Arguments:
%	  GAMMA - Gamma value for system.
%	  SIGMA2 - length scale of latent process.
%	  T1 - first time input (number of time points x 1).
%	  T2 - second time input (number of time points x 1).
%	  UPSILON - precomputation of the upsilon matrix.
%	  MODE - operation mode, according to the derivative (mode 0,
%	   derivative wrt t1, mode 1 derivative wrt t2)
%	
%
%	See also
%	LFMAACOMPUTEUPSILONMATRIX.M


%	Copyright (c) 2010 Mauricio Alvarez
% 	lfmaaGradientUpsilonMatrix.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z

sigma = sqrt(sigma2);

if nargin<6
    upsilon = lfmapComputeUpsilonMatrix(gamma, sigma2, t1, t2, 1 - mode);
    if nargin <5
        mode =0;
    end
end

dUpsi = lfmapGradientUpsilonMatrix(gamma, sigma2, t1, t2, 1 - mode);
gridt1 = repmat(t1, 1, length(t2));
gridt2 = repmat(t2', length(t1), 1);
timeGrid = gridt1 - gridt2;

if mode ==0
    dUpsilon = 2*gamma*upsilon + (gamma^2)*dUpsi + (2/(sqrt(pi)*sigma))* ...
        exp(-(timeGrid.^2)./sigma2).*(2/sigma2 - (2*timeGrid/sigma2).^2);
else
    dUpsilon = 2*gamma*upsilon + (gamma^2)*dUpsi + (2/(sqrt(pi)*sigma))* ...
        exp(-(timeGrid.^2)./sigma2).*(2/sigma2 - (2*timeGrid/sigma2).^2) ...
        + (2*gamma/(sqrt(pi)*sigma))*(exp(-gamma*t1).*(2-gamma*t1))* ...
        ((gamma-2*t2/sigma2).*exp(-t2.^2/sigma2)).' ...
        + (2*gamma^2/(sqrt(pi)*sigma))*exp(-gamma*t1)*(exp(-t2.^2/sigma2).');
end
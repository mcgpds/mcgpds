function [upsilonvp, upsilon] = lfmvpComputeUpsilonMatrix(gamma, sigma2, t1, t2, mode)

% LFMVPCOMPUTEUPSILONMATRIX Upsilon matrix vel. pos. with t1, t2 limits
%
%	Description:
%
%	UPSILON = LFMVPCOMPUTEUPSILONMATRIX(GAMMA, SIGMA2, T1, T2, MODE)
%	computes a portion of the LFM kernel.
%	 Returns:
%	  UPSILON - result of this subcomponent of the kernel for the given
%	   values.
%	 Arguments:
%	  GAMMA - Gamma value for system.
%	  SIGMA2 - length scale of latent process.
%	  T1 - first time input (number of time points x 1).
%	  T2 - second time input (number of time points x 1).
%	  MODE - operation mode, according to the derivative (mode 0,
%	   derivative wrt t1, mode 1 derivative wrt t2)
%	
%
%	See also
%	LFMCOMPUTEUPSILONMATRIX.F, LFMCOMPUTEH3.M


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	lfmvpComputeUpsilonMatrix.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z

sigma = sqrt(sigma2);
gridt1 = repmat(t1, 1, length(t2));
gridt2 = repmat(t2', length(t1), 1);
timeGrid = gridt1 - gridt2;

if mode==0
    if nargout > 1
        upsilon = lfmComputeUpsilonMatrix(gamma, sigma2, t1, t2);
        upsilonvp = -gamma*upsilon + (2/(sqrt(pi)*sigma))*exp(-(timeGrid.^2)./sigma2);
    else
        upsilonvp = -gamma*lfmComputeUpsilonMatrix(gamma, sigma2, t1, t2) ...
            + (2/(sqrt(pi)*sigma))*exp(-(timeGrid.^2)./sigma2);
    end
else
    if nargout > 1
        upsilon = lfmComputeUpsilonMatrix(gamma, sigma2, t1, t2);
        upsilonvp = gamma*upsilon - (2/(sqrt(pi)*sigma))*exp(-(timeGrid.^2)./sigma2) ...
            + (2/(sqrt(pi)*sigma))*exp(-gamma*t1)*(exp(-(t2.^2)/sigma2)).';
    else
        upsilonvp = gamma*lfmComputeUpsilonMatrix(gamma, sigma2, t1, t2) ...
            - (2/(sqrt(pi)*sigma))*exp(-(timeGrid.^2)./sigma2) ...
            + (2/(sqrt(pi)*sigma))*exp(-gamma*t1)*(exp(-(t2.^2)/sigma2)).';
    end
end


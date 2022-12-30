function dUpsilonS = lfmavGradientSigmaUpsilonMatrix(gamma, sigma2, ...
    t1, t2, mode)

% LFMAVGRADIENTSIGMAUPSILONMATRIX Gradient of upsilon matrix av wrt sigma
%
%	Description:
%
%	UPSILON = LFMAVGRADIENTSIGMAUPSILONMATRIX(GAMMA, SIGMA2, T1, T2,
%	MODE) computes the gradient wrt sigma of a portion of the LFMAV
%	kernel.
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
%	LFMAVCOMPUTEUPSILONMATRIX.M


%	Copyright (c) 2010 Mauricio Alvarez
% 	lfmavGradientSigmaUpsilonMatrix.m SVN version 807
% 	last update 2011-07-04T18:55:15.884563Z

gridt1 = repmat(t1, 1, length(t2));
gridt2 = repmat(t2', length(t1), 1);
timeGrid = gridt1 - gridt2;

dUpsilon = lfmvpGradientSigmaUpsilonMatrix(gamma, sigma2, t1, t2, 1-mode);

if mode == 0
    dUpsilonS = (gamma^2)*dUpsilon + (4/(sqrt(pi)*sigma2^2))*timeGrid.* ...
        exp(-(timeGrid.^2)./sigma2).*((gamma + 2*timeGrid/sigma2).*...
        (3 - 2*(timeGrid.^2)/sigma2) + 4*timeGrid/sigma2) ...
        - (4/(sqrt(pi)*sigma2^2))*exp(-(timeGrid.^2)./sigma2).*...
        (3 - 2*(timeGrid.^2)/sigma2);
else
    dUpsilonS = (gamma^2)*dUpsilon - (4/(sqrt(pi)*sigma2^2))*timeGrid.* ...
        exp(-(timeGrid.^2)./sigma2).*((gamma + 2*timeGrid/sigma2).*...
        (3 - 2*(timeGrid.^2)/sigma2) + 4*timeGrid/sigma2) ...
        + (4/(sqrt(pi)*sigma2^2))*exp(-(timeGrid.^2)./sigma2).*...
        (3 - 2*(timeGrid.^2)/sigma2) + (2*gamma/(sqrt(pi)*sigma2))*...
        exp(-gamma*t1)*(((gamma - 2*t2/sigma2).*(1-2*t2.^2/sigma2) - ...
        4*t2/sigma2).*exp(-t2.^2/sigma2)).';
end

function [h, compUpAA, compUpAP] =  lfmComputeH3AA(gamma1_p, gamma1_m, sigma2, t1, ...
    t2, preFactor, mode)

% LFMCOMPUTEH3AA Helper function for computing part of the LFMAA kernel.
%
%	Description:
%
%	H = LFMCOMPUTEH3AA(GAMMA1, GAMMA2, SIGMA2, T1, T2, PREFACTOR, MODE)
%	computes a portion of the LFMAA kernel.
%	 Returns:
%	  H - result of this subcomponent of the kernel for the given
%	   values.
%	 Arguments:
%	  GAMMA1 - Gamma value for first system.
%	  GAMMA2 - Gamma value for second system.
%	  SIGMA2 - length scale of latent process.
%	  T1 - first time input (number of time points x 1).
%	  T2 - second time input (number of time points x 1).
%	  PREFACTOR - precomputed constants.
%	  MODE - indicates the correct derivative.


%	Copyright (c) 2010 Mauricio Alvarez
% 	lfmComputeH3AA.m SVN version 807
% 	last update 2010-05-28T06:01:33.000000Z

% Evaluation of h

if nargout>1    
    [compUpAA{1}, compUpAP{1}] = lfmaaComputeUpsilonMatrix(gamma1_p,sigma2, t1,t2, mode);
    [compUpAA{2}, compUpAP{2}] = lfmaaComputeUpsilonMatrix(gamma1_m,sigma2, t1,t2, mode);
    h = preFactor(1)*compUpAA{1} + preFactor(2)*compUpAA{2};
else
    h = preFactor(1)*lfmaaComputeUpsilonMatrix(gamma1_p,sigma2, t1,t2, mode) ...
        + preFactor(2)*lfmaaComputeUpsilonMatrix(gamma1_m,sigma2, t1,t2, mode);
end

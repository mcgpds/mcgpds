function  [h, compUpXP, compUp] =  lfmComputeH4AV(gamma1_p, gamma1_m, sigma2, t1, ...
    preFactor, preExp, mode)

% LFMCOMPUTEH4AV Helper function for computing part of the LFMAV kernel.
%
%	Description:
%
%	H = LFMCOMPUTEH4AV(GAMMA1, GAMMA2, SIGMA2, T1, T2, MODE) computes a
%	portion of the LFMAV kernel.
%	 Returns:
%	  H - result of this subcomponent of the kernel for the given
%	   values.
%	 Arguments:
%	  GAMMA1 - Gamma value for first system.
%	  GAMMA2 - Gamma value for second system.
%	  SIGMA2 - length scale of latent process.
%	  T1 - first time input (number of time points x 1).
%	  T2 - second time input (number of time points x 1).
%	  MODE - indicates in which way the vectors t1 and t2 must be
%	   transposed
%	
%
%	See also
%	LFMCOMPUTEH4.M, LFMCOMPUTEH4AP.M


%	Copyright (c) 2010 Mauricio Alvarez
% 	lfmComputeH4AV.m SVN version 807
% 	last update 2011-06-16T07:23:44.000000Z

if mode==0
    if nargout > 1
        [compUpXP{1}, compUp{1}] = lfmapComputeUpsilonVector(gamma1_p,sigma2, t1);
        [compUpXP{2}, compUp{2}] = lfmapComputeUpsilonVector(gamma1_m,sigma2, t1);
        h =  compUpXP{1}*( preExp(:,2)/preFactor(2) - preExp(:,1)/preFactor(1)).' ...
            + compUpXP{2}*( preExp(:,1)/preFactor(4)- preExp(:,2)/preFactor(3)).';
    else
        h =  lfmapComputeUpsilonVector(gamma1_p,sigma2, t1)*(preExp(:,2)/preFactor(2) - preExp(:,1)/preFactor(1)).' ...
            + lfmapComputeUpsilonVector(gamma1_m,sigma2, t1)*(preExp(:,1)/preFactor(4) - preExp(:,2)/preFactor(3)).';
    end
else
    if nargout > 1
        [compUpXP{1}, compUp{1}] = lfmvpComputeUpsilonVector(gamma1_p,sigma2, t1);
        [compUpXP{2}, compUp{2}] = lfmvpComputeUpsilonVector(gamma1_m,sigma2, t1);
        h =  compUpXP{1}*(preExp(:,1)/preFactor(1) - preExp(:,2)/preFactor(2)).' ...
            + compUpXP{2}*(preExp(:,2)/preFactor(3) - preExp(:,1)/preFactor(4)).';
    else
        h =  lfmvpComputeUpsilonVector(gamma1_p,sigma2, t1)*(preExp(:,1)/preFactor(1) - preExp(:,2)/preFactor(2)).' ...
            + lfmvpComputeUpsilonVector(gamma1_m,sigma2, t1)*(preExp(:,2)/preFactor(3) - preExp(:,1)/preFactor(4)).';
    end
end
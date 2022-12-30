function  [h, compUpJP] =  lfmComputeH4JP(gamma1_p, gamma1_m, sigma2, t1, ...
    preFactor, preExp, mode)

% LFMCOMPUTEH4JP Helper function for computing part of the LFMJP kernel.
%
%	Description:
%
%	H = LFMCOMPUTEH4JP(GAMMA1, GAMMA2, SIGMA2, T1, T2, MODE) computes a
%	portion of the LFMAP kernel.
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
% 	lfmComputeH4JP.m SVN version 807
% 	last update 2010-05-28T06:01:33.000000Z

if mode==0
    if nargout > 1
        compUpJP{1} = lfmjpComputeUpsilonVector(gamma1_p,sigma2, t1);
        compUpJP{2} = lfmjpComputeUpsilonVector(gamma1_m,sigma2, t1);
        h =  compUpJP{1}*( preExp(:,1)/preFactor(1) - preExp(:,2)/preFactor(2)).' ...
            + compUpJP{2}*( preExp(:,2)/preFactor(3) - preExp(:,1)/preFactor(4)).';
    else
        h =  lfmjpComputeUpsilonVector(gamma1_p,sigma2, t1)*( preExp(:,1)/preFactor(1) - preExp(:,2)/preFactor(2)).' ...
            + lfmjpComputeUpsilonVector(gamma1_m,sigma2, t1)*( preExp(:,2)/preFactor(3) - preExp(:,1)/preFactor(4)).';
    end
else
    if nargout > 1
        compUp{1} = lfmComputeUpsilonVector(gamma1_p,sigma2, t1);
        compUp{2} = lfmComputeUpsilonVector(gamma1_m,sigma2, t1);
        h =  compUp{1}*(preExp(:,2)/preFactor(2) - preExp(:,1)/preFactor(1)).' ...
            + compUp{2}*(preExp(:,1)/preFactor(4) - preExp(:,2)/preFactor(3)).';
        compUpJP = compUp;
    else
        h =  lfmComputeUpsilonVector(gamma1_p,sigma2, t1)*(preExp(:,2)/preFactor(2) - preExp(:,1)/preFactor(1) ).' ...
            + lfmComputeUpsilonVector(gamma1_m,sigma2, t1)*(preExp(:,1)/preFactor(4) - preExp(:,2)/preFactor(3)).';
    end
end
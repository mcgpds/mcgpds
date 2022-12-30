function [g1, g2, g3] = sdlfmaXsdlfmKernGradientBlock(lfmKern1, lfmKern2, ...
    t1, t2, kyy, kyv, kvy, kvv, i, j, generalConst, generalConstGrad, ...
    covGrad)

% SDLFMAXSDLFMKERNGRADIENTBLOCK Gradients of the parameters in block i,j
%
%	Description:
%
%	[G1, G2, G3] = SDLFMAXSDLFMKERNGRADIENTBLOCK(LFMKERN1, LFMKERN2, T1,
%	T2, KYY, KYV, KVY, KVV, I, J, GENERALCONSTANT, GENERALCONSTGRAD,
%	COVGRAD) computes the kernel parameters gradients for the SDLFM
%	kernel function in the block specified at indeces i,j. It assumes
%	the computation for functions that system 1 describes an
%	acceleration and system 2 describes a position.
%	 Returns:
%	  G1 - gradients of parameters for the system 1
%	  G2 - gradients of parameters for the system 2
%	  G3 - gradients of switching points
%	 Arguments:
%	  LFMKERN1 - structure containing parameters for the system 1
%	  LFMKERN2 - structure containing parameters for the system 2
%	  T1 - times at which the system 1 is evaluated
%	  T2 - times at which the system 2 is evaluated
%	  KYY - covariance for the initial conditions between position 1 and
%	   position 2 at block i,j
%	  KYV - covariance for the initial conditions between position 1 and
%	   velocity 2 at block i,j
%	  KVY - covariance for the initial conditions between velocity 1 and
%	   position 2 at block i,j
%	  KVV - covariance for the initial conditions between velocity 1 and
%	   velocity 2 at block i,j
%	  I - interval to be evaluated for system 1
%	  J - interval to be evaluated for system 2
%	  GENERALCONSTANT - constants evaluated with
%	   sdlfmKernComputeConstant.m
%	  GENERALCONSTGRAD - derivatives of the constants computed with
%	   sdlfmKernGradientConstant.m
%	  COVGRAD - partial derivatives of the objective function wrt
%	   portion of the kernel matrix in block i,j


%	Copyright (c) 2010. Mauricio A. Alvarez
% 	sdlfmaXsdlfmKernGradientBlock.m SVN version 807
% 	last update 2011-07-04T18:55:18.654565Z


if nargin<11
    j = i;
    generalConst = [];
end

g3 = [];

% Compute derivatives of the mean terms with respect to the parameters

[g1Mean, g2Mean, gsp1Mean, gsp2Mean] = sdlfmKernGradientMean(lfmKern1(1), ...
    lfmKern2(1), t1, t2, kyy, kyv, kvy, kvv, covGrad, {'sdlfma','sdlfm'}, ...
    {'sdlfmj', 'sdlfmv'});

if i==j
    [g1, g2, g3t] = sdlfmaXsdlfmKernGradientBlockIEJ(lfmKern1, lfmKern2, t1, ...
        t2, covGrad, g1Mean, g2Mean, gsp1Mean, gsp2Mean, {'lfma', 'lfm'}, ...
        {'lfmj', 'lfm'}, {'lfma', 'lfmv'});
    g3(i) = g3t;
else   
    if i>j
        [g1, g2, g3] = sdlfmXsdlfmKernGradientBlockIGJ(lfmKern1, lfmKern2, ...
            t1, t2, i, j, generalConst, generalConstGrad, covGrad, g1Mean, ...
            g2Mean, gsp1Mean, gsp2Mean, 'sdlfma', 'sdlfmj', 'lfmXlfm', ...
            'lfmvXlfm', {'lfmvXlfm', 'lfmvXlfm'}, {'lfmaXlfm', 'lfmvXlfmv'});
    else        
        [g1, g2, g3] = sdlfmaXsdlfmKernGradientBlockILJ(lfmKern1, lfmKern2, ...
            t1, t2, i, j, generalConst, generalConstGrad, covGrad, g1Mean, ...
            g2Mean, gsp1Mean, gsp2Mean, 'sdlfm', 'sdlfmv', 'lfmaXlfm', ...
            'lfmaXlfmv',{'lfmjXlfm', 'lfmaXlfmv'}, {'lfmjXlfmv', 'lfmaXlfma'});
    end
end

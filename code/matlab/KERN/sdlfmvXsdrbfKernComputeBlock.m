function K = sdlfmvXsdrbfKernComputeBlock(lfmKern, rbfKern, t1, t2, ...
    i, j, generalConst)

% SDLFMVXSDRBFKERNCOMPUTEBLOCK Cross kernel between SDLFM and SDRBF for i,j
%
%	Description:
%
%	K = SDLFMVXSDRBFKERNCOMPUTEBLOCK(LFMKERN, RBFKERN, T1, T2, I, J,
%	GENERALCONSTANT) computes the kernel matrix between a SDLFM kernel
%	function and a SDRBF kernel function in the block specified at
%	indeces i,j. It assumes the computation for a function that
%	describes a velocity.
%	 Returns:
%	  K - the kernel matrix portion of block i,j
%	 Arguments:
%	  LFMKERN - structure containing parameters for the outputs system
%	  RBFKERN - structure containing parameters for the latent system
%	  T1 - times at which the system 1 is evaluated
%	  T2 - times at which the system 2 is evaluated
%	  I - interval to be evaluated for system 1
%	  J - interval to be evaluated for system 2
%	  GENERALCONSTANT - constants evaluated with
%	   sdlfmKernComputeConstant.m


%	Copyright (c) 2010. Mauricio A. Alvarez
% 	sdlfmvXsdrbfKernComputeBlock.m SVN version 807
% 	last update 2010-05-28T06:01:33.000000Z

if nargin<7
    j = i;
    generalConst = [];
end

if i==j
    K  = lfmvXrbfKernCompute(lfmKern, rbfKern, t1, t2);    
else
    g1 = sdlfmvMeanCompute(lfmKern, t1, 'Pos');
    h1 = sdlfmvMeanCompute(lfmKern, t1, 'Vel');
    if i>j
        PosRbf = lfmXrbfKernCompute(lfmKern, rbfKern, rbfKern.limit, t2);
        VelRbf = lfmvXrbfKernCompute(lfmKern, rbfKern, rbfKern.limit, t2);
        if isempty(generalConst{i,j})
            K =  g1*PosRbf + h1*VelRbf;
        else
            K = (generalConst{i,j}(1,1)*g1 + generalConst{i,j}(2,1)*h1)*PosRbf + ...
                (generalConst{i,j}(1,2)*g1 + generalConst{i,j}(2,2)*h1)*VelRbf;
        end
    end
end


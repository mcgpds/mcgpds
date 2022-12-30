function K = sdlfmaXsdlfmKernComputeBlock(lfmKern1, lfmKern2, t1, t2, ...
    kyy, kyv, kvy, kvv, i, j, generalConst)

% SDLFMAXSDLFMKERNCOMPUTEBLOCK Computes SDLFM kernel matrix for block i,j
%
%	Description:
%
%	K = SDLFMAXSDLFMKERNCOMPUTEBLOCK(LFMKERN1, LFMKERN2, T1, T2, KYY,
%	KYV, KVY, KVV, I, J, GENERALCONSTANT) computes the kernel matrix for
%	the SDLFM kernel function in the block specified at indeces i,j. It
%	assumes the computation for functions that describe acceleration
%	(system 1) and position (system 2)
%	 Returns:
%	  K - the kernel matrix portion of block i,j
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


%	Copyright (c) 2010. Mauricio A. Alvarez
% 	sdlfmaXsdlfmKernComputeBlock.m SVN version 820
% 	last update 2010-05-28T06:01:33.000000Z

if nargin<11
    j = i;
    generalConst = [];
end

a1 = sdlfmaMeanCompute(lfmKern1(1), t1, 'Pos');
b1 = sdlfmaMeanCompute(lfmKern1(1), t1, 'Vel');
c2 = sdlfmMeanCompute(lfmKern2(1), t2, 'Pos');
e2 = sdlfmMeanCompute(lfmKern2(1), t2, 'Vel');

K = kyy*a1*c2.' + kyv*a1*e2.' + kvy*b1*c2.' + kvv*b1*e2.';

if i==j
    for k=1:length(lfmKern1)
        K  = K + lfmaXlfmKernCompute(lfmKern1(k), lfmKern2(k), t1, t2);
    end
else    
    if i>j
        PosPos = zeros(1, length(t2));
        PosVel = zeros(1, length(t2));
        for k =1:length(lfmKern1)
            PosPos = PosPos + lfmXlfmKernCompute(lfmKern1(k), lfmKern2(k), lfmKern2(k).limit, t2);
            PosVel = PosVel + lfmvXlfmKernCompute(lfmKern1(k), lfmKern2(k), lfmKern2(k).limit, t2);
        end
        if isempty(generalConst{i,j})
            K = K + a1*PosPos + b1*PosVel;
        else
           
            K = K + (generalConst{i,j}(1,1)*a1 + generalConst{i,j}(2,1)*b1)*PosPos + ...
                (generalConst{i,j}(1,2)*a1 + generalConst{i,j}(2,2)*b1)*PosVel;
        end
    else
        AccelPos = zeros(length(t1), 1);
        AccelVel = zeros(length(t1), 1);
        for k =1:length(lfmKern1)
            AccelPos = AccelPos + lfmaXlfmKernCompute(lfmKern1(k), lfmKern2(k), t1, lfmKern1(k).limit);
            AccelVel = AccelVel + lfmaXlfmvKernCompute(lfmKern1(k), lfmKern2(k), t1, lfmKern1(k).limit);
        end
        if isempty(generalConst{i,j})
            K = K + AccelPos*c2.' + AccelVel*e2.';
        else
            K = K + AccelPos*(generalConst{i,j}(1,1)*c2.' + generalConst{i,j}(2,1)*e2.') + ...
                AccelVel*(generalConst{i,j}(1,2)*c2.' + generalConst{i,j}(2,2)*e2.');
        end
    end
end


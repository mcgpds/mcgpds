function K = sdlfmaXsdlfmaKernComputeBlock(lfmKern1, lfmKern2, t1, t2, ...
    kyy, kyv, kvy, kvv, i, j, generalConst)

% SDLFMAXSDLFMAKERNCOMPUTEBLOCK Computes SDLFM kernel matrix for block i,j
%
%	Description:
%
%	K = SDLFMAXSDLFMAKERNCOMPUTEBLOCK(LFMKERN1, LFMKERN2, T1, T2, KYY,
%	KYV, KVY, KVV, I, J, GENERALCONSTANT) computes the kernel matrix for
%	the SDLFM kernel function in the block specified at indeces i,j. It
%	assumes the computation for functions that describe accelerations
%	(acceleration 1 and acceleration 2).
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
% 	sdlfmaXsdlfmaKernComputeBlock.m SVN version 820
% 	last update 2011-06-16T07:23:44.000000Z

if nargin<11
    j = i;
    generalConst = [];
end

a1 = sdlfmaMeanCompute(lfmKern1(1), t1, 'Pos');
b1 = sdlfmaMeanCompute(lfmKern1(1), t1, 'Vel');
a2 = sdlfmaMeanCompute(lfmKern2(1), t2, 'Pos');
b2 = sdlfmaMeanCompute(lfmKern2(1), t2, 'Vel');

K = kyy*a1*a2.' + kyv*a1*b2.' + kvy*b1*a2.' + kvv*b1*b2.';

if i==j
    for k=1:length(lfmKern1)
        K  = K + lfmaXlfmaKernCompute(lfmKern1(k), lfmKern2(k), t1, t2);
    end
else    
    if i>j
        AccelPos = zeros(1, length(t2));
        AccelVel = zeros(1, length(t2));
        for k=1:length(lfmKern1)
            AccelPos = AccelPos + lfmaXlfmKernCompute(lfmKern2(k), lfmKern1(k), t2, lfmKern2(k).limit).'; 
            AccelVel = AccelVel + lfmaXlfmvKernCompute(lfmKern2(k), lfmKern1(k), t2, lfmKern2(k).limit).';
        end
        if isempty(generalConst{i,j})
            K = K + a1*AccelPos + b1*AccelVel;        
        else
            K = K + (generalConst{i,j}(1,1)*a1 + generalConst{i,j}(2,1)*b1)*AccelPos + ...
                (generalConst{i,j}(1,2)*a1 + generalConst{i,j}(2,2)*b1)*AccelVel;           
        end 
    else
        AccelPos = zeros(length(t1),1);
        AccelVel = zeros(length(t1),1);
        for k =1:length(lfmKern1)
            AccelPos = AccelPos + lfmaXlfmKernCompute(lfmKern1(k), lfmKern2(k), t1, lfmKern1(k).limit);
            AccelVel = AccelVel + lfmaXlfmvKernCompute(lfmKern1(k), lfmKern2(k), t1, lfmKern1(k).limit);
        end
        if isempty(generalConst{i,j})
            K = K + AccelPos*a2.' + AccelVel*b2.';
        else
            K = K + AccelPos*(generalConst{i,j}(1,1)*a2.' + generalConst{i,j}(2,1)*b2.') + ...
                AccelVel*(generalConst{i,j}(1,2)*a2.' + generalConst{i,j}(2,2)*b2.');
        end
    end
end


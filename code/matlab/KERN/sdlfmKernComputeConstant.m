function generalConst = sdlfmKernComputeConstant(nIntervals, ...
    lfmKern1, lfmKern2, spVector)

% SDLFMKERNCOMPUTECONSTANT Compute constants for the SDLFM kernel
%
%	Description:
%
%	GENERALCONSTANT = SDLFMKERNCOMPUTECONSTANT(NINTERVALS, LFMKERN1,
%	LFMKERN2, SPVECTOR) computes necessary constants in order to compute
%	the SDLFM kernel matrix.
%	 Returns:
%	  GENERALCONSTANT - a cell containing the necessary constants for
%	   computing the kernel in the switching intervals.
%	 Arguments:
%	  NINTERVALS - number of switching intervals in the kernel
%	  LFMKERN1 - structure containing the parameters of system 1
%	  LFMKERN2 - structure containing the parameters of system 2
%	  SPVECTOR - vector containing the switching time values


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	sdlfmKernComputeConstant.m SVN version 807
% 	last update 2010-05-28T06:01:33.000000Z

generalConst = cell(nIntervals);

for i=1:nIntervals
    for j =1:i-1
        if i - j>=2
            c1 = sdlfmMeanCompute(lfmKern1, spVector(i) - spVector(i-1), 'Pos');
            e1 = sdlfmMeanCompute(lfmKern1, spVector(i) - spVector(i-1), 'Vel');
            c2 = sdlfmMeanCompute(lfmKern2, spVector(i) - spVector(i-1), 'Pos');
            e2 = sdlfmMeanCompute(lfmKern2, spVector(i) - spVector(i-1), 'Vel');
            g1 = sdlfmvMeanCompute(lfmKern1, spVector(i) - spVector(i-1), 'Pos');
            h1 = sdlfmvMeanCompute(lfmKern1, spVector(i) - spVector(i-1), 'Vel');
            g2 = sdlfmvMeanCompute(lfmKern2, spVector(i) - spVector(i-1), 'Pos');
            h2 = sdlfmvMeanCompute(lfmKern2, spVector(i) - spVector(i-1), 'Vel');
            if i - j == 2
                generalConst{i,j}(1,1) = c1; generalConst{i,j}(1,2) = e1;
                generalConst{i,j}(2,1) = g1; generalConst{i,j}(2,2) = h1;
                generalConst{j,i}(1,1) = c2; generalConst{j,i}(1,2) = e2;
                generalConst{j,i}(2,1) = g2; generalConst{j,i}(2,2) = h2;
            else
                generalConst{i,j}(1,1) = c1*generalConst{i-1,j}(1,1) + e1*generalConst{i-1,j}(2,1);
                generalConst{i,j}(1,2) = c1*generalConst{i-1,j}(1,2) + e1*generalConst{i-1,j}(2,2);
                generalConst{i,j}(2,1) = g1*generalConst{i-1,j}(1,1) + h1*generalConst{i-1,j}(2,1);
                generalConst{i,j}(2,2) = g1*generalConst{i-1,j}(1,2) + h1*generalConst{i-1,j}(2,2);
                generalConst{j,i}(1,1) = c2*generalConst{j,i-1}(1,1) + e2*generalConst{j,i-1}(2,1);
                generalConst{j,i}(1,2) = c2*generalConst{j,i-1}(1,2) + e2*generalConst{j,i-1}(2,2);
                generalConst{j,i}(2,1) = g2*generalConst{j,i-1}(1,1) + h2*generalConst{j,i-1}(2,1);
                generalConst{j,i}(2,2) = g2*generalConst{j,i-1}(1,2) + h2*generalConst{j,i-1}(2,2);
            end
        end
    end
end


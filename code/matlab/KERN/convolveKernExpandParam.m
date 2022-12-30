function kern = convolveKernExpandParam(kern,params)

%	Copyright (c) 2013 ZhaoJing
% 	last update 2013-07-29
startVal=1;
endVal=kern.outputDimension;
kern.S=params(1,startVal:endVal)';
startVal=endVal+1;
endVal=endVal+kern.inputDimension*kern.outputDimension;
kern.P_d=reshape(params(1,startVal:endVal),kern.outputDimension,kern.inputDimension);
startVal=endVal+1;
endVal=endVal+kern.inputDimension;
kern.Lambda_k=params(1,startVal:endVal);


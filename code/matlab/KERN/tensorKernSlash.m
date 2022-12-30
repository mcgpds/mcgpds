function kern = tensorKernSlash(kern, index)

% TENSORKERNSLASH Tensor kernel created by removing ith component.
%
%	Description:
%	kern = tensorKernSlash(kern, index)
%% 	tensorKernSlash.m CVS version 1.2
% 	tensorKernSlash.m SVN version 1
% 	last update 2011-06-16T07:23:44.000000Z

kern.nParams = kern.nParams - kern.comp{index}.nParams;
kern.comp = kern.comp([1:index-1 index+1:end]);

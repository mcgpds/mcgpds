function kern = tensorKernSetIndex(kern, component, indices)

% TENSORKERNSETINDEX Set the indices in the tensor kernel.
%
%	Description:
%	kern = tensorKernSetIndex(kern, component, indices)
%% 	tensorKernSetIndex.m CVS version 1.1
% 	tensorKernSetIndex.m SVN version 1
% 	last update 2011-06-16T07:23:44.000000Z

kern = cmpndKernSetIndex(kern, component, indices);
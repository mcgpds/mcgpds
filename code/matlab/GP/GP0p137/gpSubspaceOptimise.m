function model = gpSubspaceOptimise(model,varargin)

% GPSUBSPACEOPTIMISE
%
%	Description:
%	


%	Copyright (c) 2008 Carl Henrik Ek
% 	gpSubspaceOptimise.m SVN version 105
% 	last update 2008-10-11T18:56:41.000000Z

model = gpOptimise(model,varargin{:});

return;
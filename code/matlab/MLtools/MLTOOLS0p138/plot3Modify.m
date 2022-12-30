function handle = plot3Modify(handle, values, Y)

% PLOT3MODIFY Helper code for visualisation of 3-d data.
%
%	Description:
%	handle = plot3Modify(handle, values, Y)
%% 	plot3Modify.m SVN version 1414
% 	last update 2011-02-16T17:49:36.000000Z

set(handle, 'XData', values(1), 'YData', values(2), 'ZData', values(3));

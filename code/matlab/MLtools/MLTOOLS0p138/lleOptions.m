function options = lleOptions(neighbours)

% LLEOPTIONS Options for a locally linear embedding.
%
%	Description:
%
%	OPTIONS = LLEOPTIONS(NEIGHBOURS) returns the default options for a
%	locally linear embedding.
%	 Returns:
%	  OPTIONS - default options structure for locally linear embedding.
%	 Arguments:
%	  NEIGHBOURS - the number of neighbours to use.
%	
%
%	See also
%	LLECREATE, MODELCREATE


%	Copyright (c) 2008, 2009 Neil D. Lawrence
% 	lleOptions.m SVN version 1233
% 	last update 2010-12-08T16:51:59.000000Z

  if nargin < 1
    neighbours = 7;
  end
  options.numNeighbours = neighbours;
  options.isNormalised = true;
  options.regulariser = 0.0;
  options.acyclic = false;
end
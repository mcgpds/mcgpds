function kern = diagKernExpandParam(kern, params)

% DIAGKERNEXPANDPARAM Create kernel structure from DIAG kernel's parameters.
%
%	Description:
%
%	KERN = DIAGKERNEXPANDPARAM(KERN, PARAM) returns a diagonal noise
%	covariance function kernel structure filled with the parameters in
%	the given vector. This is used as a helper function to enable
%	parameters to be optimised in, for example, the NETLAB optimisation
%	functions.
%	 Returns:
%	  KERN - kernel structure with the given parameters in the relevant
%	   locations.
%	 Arguments:
%	  KERN - the kernel structure in which the parameters are to be
%	   placed.
%	  PARAM - vector of parameters which are to be placed in the kernel
%	   structure.
%	
%
%	See also
%	DIAGKERNPARAMINIT, DIAGKERNEXTRACTPARAM, KERNEXPANDPARAM


%	Copyright (c) 2011 Neil D. Lawrence
% 	diagKernExpandParam.m SVN version 1566
% 	last update 2011-07-31T10:48:49.890337Z

kern.variance = params(1);
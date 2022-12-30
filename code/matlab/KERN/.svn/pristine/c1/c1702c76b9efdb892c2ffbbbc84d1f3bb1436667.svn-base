function kern = nddisimKernExpandParamTransformSettings(kern, paramtransformsettings)

% NDDISIMKERNEXPANDPARAMTRANSFORMSETTINGS Create kernel structure from DISIM kernel's parameters.
%
%	Description:
%
%	KERN = NDDISIMKERNEXPANDPARAMTRANSFORMSETTINGS(KERN, PARAM) returns
%	a single input motif kernel structure filled with the parameters in
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
%	
%
%	See also
%	DISIMKERNPARAMINIT, DISIMKERNEXTRACTPARAM, KERNEXPANDPARAM


%	Copyright (c) 2006 Neil D. Lawrence
%	Copyright (c) 2007-2009 Antti Honkela
%	Copyright (c) 2011 Jaakko Peltonen
% 	nddisimKernExpandParamTransformSettings.m SVN version 1567
% 	last update 2011-08-09T18:31:55.769804Z

for k=1:5,
  kern.transforms(k).transformsettings = paramtransformsettings{k};
end;



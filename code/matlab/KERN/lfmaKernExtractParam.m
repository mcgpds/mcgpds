function [params, names] = lfmaKernExtractParam(kern)

% LFMAKERNEXTRACTPARAM Extract parameters from the LFMA kernel structure.
%
%	Description:
%
%	PARAM = LFMAKERNEXTRACTPARAM(KERN) Extract parameters from the LFMA
%	kernel structure into a vector of parameters for optimisation.
%	 Returns:
%	  PARAM - vector of parameters extracted from the kernel.
%	 Arguments:
%	  KERN - the kernel structure containing the parameters to be
%	   extracted.
%
%	[PARAM, NAMES] = LFMAKERNEXTRACTPARAM(KERN) Extract parameters from
%	the LFMA kernel structure into a vector of parameters for
%	optimisation.
%	 Returns:
%	  PARAM - vector of parameters extracted from the kernel.
%	  NAMES - cell array of strings containing parameter names.
%	 Arguments:
%	  KERN - the kernel structure containing the parameters to be
%	   extracted.
%	
%
%	See also
%	% SEEALSO LFMKERNPARAMINIT, LFMKERNEXPANDPARAM, KERNEXTRACTPARAM, SCG, CONJGRAD


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	lfmaKernExtractParam.m SVN version 809
% 	last update 2010-05-28T06:01:33.000000Z

[params, names] = lfmKernExtractParam(kern);
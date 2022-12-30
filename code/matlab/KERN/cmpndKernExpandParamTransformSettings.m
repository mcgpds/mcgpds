function kern = cmpndKernExpandParamTransformSettings(kern, paramtransformsettings)

% CMPNDKERNEXPANDPARAMTRANSFORMSETTINGS Create kernel structure from CMPND kernel's parameter transformation settings.
%
%	Description:
%	
%
%	CMPNDKERNEXPANDPARAMTRANSFORMSETTINGS
%	DESC returns a compound kernel structure filled with the
%	parameter transformation settings in the given cell array. This
%	is used as a helper function to enable parameters to be optimised
%	in, for example, the NETLAB optimisation functions.
%	
%	ARG kern : the kernel structure in which the parameters are to be
%	placed.
%	
%	ARG param : cell array of parameter transformation settings which
%	are to be placed in the kernel structure. Each setting needs to
%	correspond to the transformations used by the kernel. For
%	example, some transformation might need knowledge of the desired
%	output range, so the transformation setting could be e.g. [0 100],
%	whereas some other transformations might need more complicated
%	setting information.
%	
%	RETURN kern : kernel structure with the given parameter
%	transformation settings in the relevant locations.
%	
%	
%	
%
%	See also
%	CMPNDKERNPARAMINIT, CMPNDKERNEXTRACTPARAM, KERNEXPANDPARAM


%	Copyright (c) 2004, 2005, 2006 Neil D. Lawrence
%	Copyright (c) 2011 Jaakko Peltonen
% 	cmpndKernExpandParamTransformSettings.m SVN version 1567
% 	last update 2011-08-09T18:29:09.559803Z

%fprintf(1,'cmpndKernExpandParam, params, kern.paramGroups\n')
%params
%full(kern.paramGroups)


% There should be as many transformation settings provided as there
% are shared parameters in the compound kernel.
if length(paramtransformationsettings)~=size(kern.paramGroups,2),
  error(sprintf('Problem in cmpndKernExpandParamTransformSettings: expected %d transformation-settings, received %d\n',size(kern.paramGroups,2),length(paramtransformationsettings)));
end;


% In a compound (cmpnd) kernel, some parameters may be shared
% between groups. Each provided transformation-setting is 
% assumed to be specific to a single shared parameter, so 
% the transformation-setting has to be duplicated for each group
% using that shared parameter.

expandedsettings=cell(size(kern.paramGroups,1),1);
for k=1:length(expandedsettings),
  % Find the shared parameter that correspond to the k:th non-shared parameter
  k2=find(kern.paramGroups(k,:)==1);
  
  expandedsettings{k}=paramtransformsettings{k2};
end;


%fprintf(1,'expanding CMPND kernel param-transforms settings:\n');
%paramtransformsettings
%expandedsettings
%pause


% Distribute each group of transformation-settings into the
% corresponding component-kernel of the compound kernel.
startVal = 1;
endVal = 0;
for i = 1:length(kern.comp)
  % Each component-kernel is assumed to require as many transformation
  % settings as it has parameters.
  endVal = endVal + kern.comp{i}.nParams;
  kern.comp{i} = kernExpandParamTransformSettings(kern.comp{i}, expandedsettings(startVal:endVal));
  startVal = endVal + 1;
end

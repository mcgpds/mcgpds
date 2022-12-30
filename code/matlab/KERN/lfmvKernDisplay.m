function lfmvKernDisplay(varargin)

% LFMVKERNDISPLAY Display parameters of the LFMV kernel.
%
%	Description:
%
%	LFMVKERNDISPLAY(KERN) displays the parameters of the LFMV kernel and
%	the kernel type to the console.
%	 Arguments:
%	  KERN - the kernel to display.
%
%	LFMVKERNDISPLAY(KERN, SPACING)
%	 Arguments:
%	  KERN - the kernel to display.
%	  SPACING - how many spaces to indent the display of the kernel by.
%	
%
%	See also
%	LFMKERNPARAMINIT, MODELDISPLAY, KERNDISPLAY


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	lfmvKernDisplay.m SVN version 809
% 	last update 2011-06-16T07:23:44.000000Z


lfmKernDisplay(varargin{:});
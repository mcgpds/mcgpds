function [k, sk] = conwhiteKernCompute(kern, x, x2)

% WHITEKERNCOMPUTE Compute the white-noise (WHITE) kernel between
%
%	Description:
%	two sets of input locations x and x2, given the kernel
%	parameters. A special option is allowed for single-dimensional
%	inputs, where variances are diagonal but non-isotropic and are
%	interpolated between fixed variances at several locations.
%	
%
%	WHITEKERNCOMPUTE computes the kernel parameters for the white noise
%	kernel given inputs associated with rows and columns.
%	ARG kern : the kernel structure for which the matrix is computed.
%	ARG x : the input matrix associated with the rows of the kernel.
%	ARG x2 : the inpute matrix associated with the columns of the kernel.
%	RETURN k : the kernel matrix computed at the given points.
%	
%
%	K = WHITEKERNCOMPUTE(KERN, X) computes the kernel matrix for the
%	white noise kernel given a design matrix of inputs.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X - input data matrix in the form of a design matrix.
%	
%	
%
%	See also
%	WHITEKERNPARAMINIT, KERNCOMPUTE, KERNCREATE, WHITEKERNDIAGCOMPUTE


%	Copyright (c) 2004, 2005, 2006, 2009 Neil D. Lawrence
%	Copyright (c) 2011 Jaakko Peltonen
%	Copyright (c) 2013 ZhaoJing
% 	whiteKernCompute.m CVS version 1.4
% 	whiteKernCompute.m SVN version 1534
% 	last update 2011-08-03T14:11:43.315503Z

if nargin < 3
  
  if (isfield(kern,'use_fixedvariance')==1) && (kern.use_fixedvariance==1),
    % Use a fixed variance function, for single-dimensional inputs:
    % simply interpolate between given fixed values. Assume
    % kern.fixedvariance_times has been sorted in ascending order.

    % fprintf(1,'using fixed variance\n');
    tempdiag=0*x;
    for k=1:length(x),
      % Interpolate the variance for this point. First find the
      % nearest smaller and larger locations of the fixed variance
      % values.      
      I=find(kern.fixedvariance_times<=x(k)); 
      if length(I)==0, 
	% The input location x is < all of the locations of fixed
        % variance, so just take the fixed variance at the smallest
        % location.	
	% fprintf(1,'k=%d, x smaller than all\n',k);
        tempdiag(k)=kern.fixedvariance(1);
      elseif length(I)==length(kern.fixedvariance_times), 
	% The input location x is >= all of the locations of fixed
	% variance, so just take the fixed variance at the largest
	% location.
	% fprintf(1,'k=%d, x larger than all\n',k);
        tempdiag(k)=kern.fixedvariance(end);
      else
	% The input location x is between two locations of fixed
	% variance, so interpolate between those two variance values.
        ind1=max(I); % closest timepoint <= x
        I=find(kern.fixedvariance_times>x(k)); 
        ind2=min(I); % closest timepoint > x
        % fprintf(1,'k=%d, x %f, ind1 %d, ind2 %d\n',k, x(k),ind1,ind2);
        tempdiag(k) = kern.fixedvariance(ind1)+(x(k)-kern.fixedvariance_times(ind1))/(kern.fixedvariance_times(ind2)-kern.fixedvariance_times(ind1))*(kern.fixedvariance(ind2)-kern.fixedvariance(ind1));
      end;
    end;
    
    sk = diag(tempdiag);
    k = sk;
  else    
    % Just compute a kernel normally: a sparse diagonal kernel with
    % the same constant value at each diagonal entry, according to
    % the kernel's variance parameter.    
    sk = speye(size(x, 1));
    k = kern.variance*sk;
  end;
else
  % Compute the white-noise kernel between two different sets of
  % locations. Because each observations is assumed to have an independent
  % observation noise variable (even if two observations are made
  % at the same input location), the white-noise kernel is simply
  % zero between any two different observations. This happens
  % regardless of whether the variances are given by a kernel
  % parameter or whether they are interpolated between some fixed values.
    
  if (isfield(kern,'use_fixedvariance')==1) && (kern.use_fixedvariance==1),
    % fprintf(1,'using fixed variance, 2\n');
    k = spalloc(size(x, 1), size(x2, 1), 0);    
  else
    k = spalloc(size(x, 1), size(x2, 1), 0);
  end;
end

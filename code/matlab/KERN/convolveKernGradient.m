function g = convolveKernGradient(kern, x, varargin)

% RBFARD2KERNGRADIENT Gradient of RBFARD2 kernel's parameters.
%
%	Description:
%
%	G = RBFARD2KERNGRADIENT(KERN, X, PARTIAL) computes the gradient of
%	functions with respect to the automatic relevance determination
%	radial basis function kernel's parameters. As well as the kernel
%	structure and the input positions, the user provides a matrix
%	PARTIAL which gives the partial derivatives of the function with
%	respect to the relevant elements of the kernel matrix.
%	 Returns:
%	  G - gradients of the function of interest with respect to the
%	   kernel parameters. The ordering of the vector should match that
%	   provided by the function kernExtractParam.
%	 Arguments:
%	  KERN - the kernel structure for which the gradients are being
%	   computed.
%	  X - the input locations for which the gradients are being
%	   computed.
%	  PARTIAL - matrix of partial derivatives of the function of
%	   interest with respect to the kernel matrix. The argument takes the
%	   form of a square matrix of dimension  numData, where numData is
%	   the number of rows in X.
%
%	G = RBFARD2KERNGRADIENT(KERN, X1, X2, PARTIAL) computes the
%	derivatives as above, but input locations are now provided in two
%	matrices associated with rows and columns of the kernel matrix.
%	 Returns:
%	  G - gradients of the function of interest with respect to the
%	   kernel parameters.
%	 Arguments:
%	  KERN - the kernel structure for which the gradients are being
%	   computed.
%	  X1 - the input locations associated with the rows of the kernel
%	   matrix.
%	  X2 - the input locations associated with the columns of the kernel
%	   matrix.
%	  PARTIAL - matrix of partial derivatives of the function of
%	   interest with respect to the kernel matrix. The matrix should have
%	   the same number of rows as X1 and the same number of columns as X2
%	   has rows.
%	
%	
%
%	See also
%	% SEEALSO RBFARD2KERNPARAMINIT, KERNGRADIENT, RBFARD2KERNDIAGGRADIENT, KERNGRADX


%	Copyright (c) 2004, 2005, 2006 Neil D. Lawrence
%	Copyright (c) 2009 Michalis K. Titsias
%   Copyright (c) 2013 ZhaoJing
% 	convolveKernGradient.m SVN version 582
% 	last update 2009-11-08T13:06:36.000000Z



A=1./(kern.Lambda_k);
if nargin < 4
  [k, dist2xx] = convolveKernCompute(kern, x);
else
  [k, dist2xx] = convolveKernCompute(kern, x, varargin{1});
end
covGradK = varargin{end}.*k;

if nargin < 4
  for i = 1:size(x, 2)
    galpha(i)  =  -(sum(covGradK*(x(:, i).*x(:, i))) ...
                   -x(:, i)'*covGradK*x(:, i));
  end
else
  for i = 1:size(x, 2)
    galpha(i) = -(0.5*sum(covGradK'*(x(:, i).*x(:, i))) ...
                 + 0.5*sum(covGradK*(varargin{1}(:, i).*varargin{1}(:, i)))...
                 -x(:, i)'*covGradK*varargin{1}(:, i));
  end
end
gLambda_k=-A.^2.*galpha;
gP_d=zeros(size(kern.P_d));

gS_d=zeros(size(kern.S));
g=[gS_d' gP_d(:)' gLambda_k];
% g(1:size(kern.S,2)) =  zeros(1,size(kern.S,2));

% if nargin < 4
%   for i = 1:size(x, 2)
%     g(size(kern.S,2) + i)  =  -(sum(covGradK*(x(:, i).*x(:, i))) ...
%                    -x(:, i)'*covGradK*x(:, i));
%   end
% else
%   for i = 1:size(x, 2)
%     g(1 + i) = -(0.5*sum(covGradK'*(x(:, i).*x(:, i))) ...
%                  + 0.5*sum(covGradK*(varargin{1}(:, i).*varargin{1}(:, i)))...
%                  -x(:, i)'*covGradK*varargin{1}(:, i));
%   end
% end





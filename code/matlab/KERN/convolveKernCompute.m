function [k, n2] = convolveKernCompute(kern, x, x2)



%	Copyright (c) 2013 ZhaoJing
% 	last update 2013-07-29
scales = sparse(diag(sqrt(1./kern.Lambda_k)));
x = x*scales;

if nargin < 3
  n2 = dist2(x, x);
  k = exp(-n2*0.5);
else
  x2 = x2*scales;
  n2 = dist2(x, x2);
  k = exp(-n2*0.5);
end
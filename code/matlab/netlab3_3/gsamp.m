function x = gsamp(mu, covar, nsamp)
%GSAMP	Sample from a Gaussian distribution.
%
%	Description
%
%	X = GSAMP(MU, COVAR, NSAMP) generates a sample of size NSAMP from a
%	D-dimensional Gaussian distribution. The Gaussian density has mean
%	vector MU and covariance matrix COVAR, and the matrix X has NSAMP
%	rows in which each row represents a D-dimensional sample vector.
%
%	See also
%	GAUSS, DEMGAUSS
%

%	Copyright (c) Ian T Nabney (1996-2001)
%  modified by ZhaoJing 2014-09-23

const = 1000; % applied on covar;

d = size(covar, 1);

mu = reshape(mu, 1, d);   % Ensure that mu is a row vector

[evec, eval] = eig(covar);

deig=diag(eval);

if (~isreal(deig)) | any(deig<0), 
  if det(covar) < 0 %% added a judge conditon
      warning('Covariance Matrix is not OK, redefined to be positive definite');
      eval=abs(eval);
  else
      % If the covar is actually positive definite but factorized with unreal eigenvalues due to the numerical solution
      % We multiply a const on the 'covar' and divide the const on the
      % eval. Note that the eigenvectors are the same. 
      covar = covar * const;
      [evec, eval] = eig(covar);
      eval = eval/const;
  end
end

coeffs = randn(nsamp, d)*sqrt(eval);

x = ones(nsamp, 1)*mu + coeffs*evec';

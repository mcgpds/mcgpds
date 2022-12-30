function K = rbfhKernCompute(kern, x1, x2)

% RBFHKERNCOMPUTE Compute the RBFH kernel given the parameters and X.
%
%	Description:
%
%	K = RBFHKERNCOMPUTE(KERN, X1, X2) computes the kernel parameters for
%	the radial basis function heat kernel given inputs associated with
%	rows and columns.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X1 - the input matrix associated with the rows of the kernel.
%	  X2 - the input matrix associated with the columns of the kernel.
%
%	K = RBFHKERNCOMPUTE(KERN, X1) computes the kernel matrix for the
%	radial basis function heat kernel given a design matrix of inputs.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X1 - input data matrix in the form of a design matrix.
%	
%
%	See also
%	RBFHKERNPARAMINIT, KERNCOMPUTE, 


%	Copyright (c) 2010 Mauricio A. Alvarez
% 	rbfhKernCompute.m SVN version 1234
% 	last update 2011-06-16T07:23:44.000000Z

if nargin < 3
    x2 = x1;
end

if size(x1, 2) ~= 2 || size(x2, 2) ~= 2
    error('Input can only have two columns');
end


% Split the domain into time domain and spatial domain and account for
% missing values. If there are no missing values the computation of the
% kernel is a pointwise prodruct, otherwise it is a kronecker product.
t1 = x1(x1(:,1)~=Inf,1);
t2 = x2(x2(:,1)~=Inf,1);
s1 = x1(x1(:,2)~=Inf,2);
s2 = x2(x2(:,2)~=Inf,2);
if (length(t1) == length(s1)) && (length(t2) == length(s2))
    ut1 = unique(t1);
    ut2 = unique(t2);
    us1 = unique(s1);
    us2 = unique(s2);
    if (length(ut1)*length(us1) == length(t1)) && ...
        (length(ut2)*length(us2) == length(t2))
        t1 = ut1; s1 = us1; t2 = ut2; s2 = us2;
        isPointwise = false;        
    else
        isPointwise = true;
    end
else
    isPointwise = false;
end

kern.rbf.inverseWidth = kern.inverseWidthTime;
Kt = rbfKernCompute(kern.rbf, t1, t2);
kern.rbf.inverseWidth = kern.inverseWidthSpace;
Ks = rbfKernCompute(kern.rbf, s1, s2);
 
if isPointwise
   K = Kt.*Ks;
else
   K = kron(Kt, Ks);    
end




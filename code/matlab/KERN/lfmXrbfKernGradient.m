function [g1, g2] = lfmXrbfKernGradient(lfmKern, rbfKern, t1, t2, covGrad, meanVector)

% LFMXRBFKERNGRADIENT Compute gradient between the LFM and RBF kernels.
%
%	Description:
%
%	[G1, G2] = LFMXRBFKERNGRADIENT(LFMKERN, RBFKERN, T) computes the
%	gradient of an objective function with respect to cross kernel terms
%	between LFM and RBF kernels for the multiple output kernel.
%	 Returns:
%	  G1 - gradient of objective function with respect to kernel
%	   parameters of LFM kernel.
%	  G2 - gradient of objective function with respect to kernel
%	   parameters of RBF kernel.
%	 Arguments:
%	  LFMKERN - the kernel structure associated with the LFM kernel.
%	  RBFKERN - the kernel structure associated with the RBF kernel.
%	  T - inputs for which kernel is to be computed.
%
%	[G1, G2] = LFMXRBFKERNGRADIENT(LFMKERN, RBFKERN, T1, T2) computes
%	the gradient of an objective function with respect to cross kernel
%	terms between LFM and RBF kernels for the multiple output kernel.
%	 Returns:
%	  G1 - gradient of objective function with respect to kernel
%	   parameters of LFM kernel.
%	  G2 - gradient of objective function with respect to kernel
%	   parameters of RBF kernel.
%	 Arguments:
%	  LFMKERN - the kernel structure associated with the LFM kernel.
%	  RBFKERN - the kernel structure associated with the RBF kernel.
%	  T1 - row inputs for which kernel is to be computed.
%	  T2 - column inputs for which kernel is to be computed.
%
%	[G1, G2] = LFMXRBFKERNGRADIENT(LFMKERN, RBFKERN, T1, T2, MEANVEC)
%	computes the gradient of an objective function with respect to cross
%	kernel terms between LFM and RBF kernels for the multiple output
%	kernel.
%	 Returns:
%	  G1 - gradient of objective function with respect to kernel
%	   parameters of LFM kernel.
%	  G2 - gradient of objective function with respect to kernel
%	   parameters of RBF kernel.
%	 Arguments:
%	  LFMKERN - the kernel structure associated with the LFM kernel.
%	  RBFKERN - the kernel structure associated with the RBF kernel.
%	  T1 - row inputs for which kernel is to be computed.
%	  T2 - column inputs for which kernel is to be computed.
%	  MEANVEC - precomputed factor that is used for the switching
%	   dynamical latent force model.
%	
%	
%	
%
%	See also
%	MULTIKERNPARAMINIT, MULTIKERNCOMPUTE, LFMKERNPARAMINIT, RBFKERNPARAMINIT


%	Copyright (c) 2007, 2008 David Luengo


%	With modifications by Neil D. Lawrence 2007


%	With modifications by Mauricio A. Alvarez 2008, 2010
% 	lfmXrbfKernGradient.m SVN version 827
% 	last update 2010-05-31T19:18:48.000000Z

subComponent = false; % This is just a flag that indicates if this kernel is part of a bigger kernel (SDLFM)

if nargin == 4
    covGrad = t2;
    t2 = t1;
elseif nargin == 6
    subComponent = true;
    if numel(meanVector)>1
        if size(meanVector,1) == 1,
            if size(meanVector, 2)~=size(covGrad, 2)
                error('The dimensions of meanVector don''t correspond to the dimensions of covGrad')
            end
        else
            if size((meanVector'), 2)~=size(covGrad,2)
                error('The dimensions of meanVector don''t correspond to the dimensions of covGrad')
            end
        end
    else
        if numel(t1)==1 && numel(t2)>1
            % matGrad will be row vector and so should be covGrad
            dimcovGrad = length(covGrad);
            covGrad = reshape(covGrad, [1 dimcovGrad]);
        elseif numel(t1)>1 && numel(t2)==1
            % matGrad will be column vector and sp should be covGrad
            dimcovGrad = length(covGrad);
            covGrad = reshape(covGrad, [dimcovGrad 1]);
        end
    end
end

if size(t1, 2) > 1 || size(t2, 2) > 1
    error('Input can only have one column');
end
if lfmKern.inverseWidth ~= rbfKern.inverseWidth
    error('Kernels cannot be cross combined if they have different inverse widths.')
end

if lfmKern.isNormalised ~= rbfKern.isNormalised
  error('Kernels cannot be cross combined if they have different normalization settings.')
end

m = lfmKern.mass;
D = lfmKern.spring;
C = lfmKern.damper;
S = lfmKern.sensitivity;

alpha = C/(2*m);
omega = sqrt(D/m-alpha^2);

sigma2 = 2/lfmKern.inverseWidth;
sigma = sqrt(sigma2);

if isreal(omega)
    gamma = alpha + j*omega;
    ComputeUpsilon1 = lfmComputeUpsilonMatrix(gamma,sigma2,t1, t2);
    if lfmKern.isNormalised
        K0 = S/(2*sqrt(2)*m*omega);
    else
        K0 = sigma*sqrt(pi)*S/(2*m*omega);
    end
else
    gamma1 = alpha + j*omega;
    gamma2 = alpha - j*omega;
    ComputeUpsilon1 = lfmComputeUpsilonMatrix(gamma2,sigma2,t1, t2);
    ComputeUpsilon2 = lfmComputeUpsilonMatrix(gamma1,sigma2,t1, t2);
    if lfmKern.isNormalised
        K0 = (S/(j*4*sqrt(2)*m*omega));
    else
        K0 = sigma*sqrt(pi)*S/(j*4*m*omega);
    end
end

g1 = zeros(1,5);

% Gradient with respect to m, C and D

for ind = 1:3 % Parameter (m, D or C)
    switch ind
        case 1  % Gradient wrt m
            gradThetaM = 1;
            gradThetaAlpha = -C/(2*(m^2));
            gradThetaOmega = (C^2-2*m*D)/(2*(m^2)*sqrt(4*m*D-C^2));
        case 2  % Gradient wrt D
            gradThetaM = 0;
            gradThetaAlpha = 0;
            gradThetaOmega = 1/sqrt(4*m*D-C^2);
        case 3  % Gradient wrt C
            gradThetaM = 0;
            gradThetaAlpha = 1/(2*m);
            gradThetaOmega = -C/(2*m*sqrt(4*m*D-C^2));
    end
    
    % Gradient evaluation
    
    if isreal(omega)
        gamma = alpha + j*omega;
        gradThetaGamma = gradThetaAlpha + j*gradThetaOmega;
        matGrad = -K0*imag(lfmGradientUpsilonMatrix(gamma,sigma2,t1, t2)*gradThetaGamma ...
            - (gradThetaM/m + gradThetaOmega/omega) ...
            * ComputeUpsilon1);
    else
        gamma1 = alpha + j*omega;
        gamma2 = alpha - j*omega;
        gradThetaGamma1 = gradThetaAlpha + j*gradThetaOmega;
        gradThetaGamma2 = gradThetaAlpha - j*gradThetaOmega;
        matGrad = K0*(lfmGradientUpsilonMatrix(gamma2,sigma2, t1, t2)*gradThetaGamma2 ...
            -  lfmGradientUpsilonMatrix(gamma1,sigma2, t1, t2)*gradThetaGamma1 ...
            - (gradThetaM/lfmKern.mass + gradThetaOmega/omega) ...
            * (ComputeUpsilon1 - ComputeUpsilon2));
        
    end
    if subComponent
        if size(meanVector,1) ==1,
            matGrad = matGrad*meanVector;
        else
            matGrad = (meanVector*matGrad).';
        end
    end
    g1(ind) = sum(sum(matGrad.*covGrad));
end

% Gradient with respect to sigma

if isreal(omega)
    gamma = alpha + j*omega;
    if lfmKern.isNormalised
        matGrad = -K0*imag(lfmGradientSigmaUpsilonMatrix(gamma,sigma2,t1, t2));
    else
        matGrad = -(sqrt(pi)*S/(2*m*omega)) ...
            * imag(ComputeUpsilon1 ...
            + sigma*lfmGradientSigmaUpsilonMatrix(gamma,sigma2,t1, t2));
    end
else
    gamma1 = alpha + j*omega;
    gamma2 = alpha - j*omega;
    if lfmKern.isNormalised
        matGrad = K0*(lfmGradientSigmaUpsilonMatrix(gamma2,sigma2,t1,t2) ...
            - lfmGradientSigmaUpsilonMatrix(gamma1,sigma2,t1,t2));
    else
        matGrad = (sqrt(pi)*S/(j*4*m*omega)) ...
            *(ComputeUpsilon1 - ComputeUpsilon2 ...
            + sigma*(lfmGradientSigmaUpsilonMatrix(gamma2,sigma2,t1,t2) ...
            - lfmGradientSigmaUpsilonMatrix(gamma1,sigma2,t1,t2)));
    end
end;

if subComponent
    if size(meanVector,1) ==1,
        matGrad = matGrad*meanVector;
    else
        matGrad = (meanVector*matGrad).';
    end
end

g1(4) = sum(sum(matGrad.*covGrad))*(-(sigma^3)/4); % temporarly introduced by MA
g2(1) = g1(4);

% Gradient with respect to S

if isreal(omega)
    if lfmKern.isNormalised
        matGrad = -(1/(2*sqrt(2)*m*omega))* imag(ComputeUpsilon1);
    else
        matGrad = -(sqrt(pi)*sigma/(2*m*omega))* imag(ComputeUpsilon1);
    end
else
    if lfmKern.isNormalised
        matGrad = (1/(j*4*sqrt(2)*m*omega))*(ComputeUpsilon1 - ComputeUpsilon2);
    else
        matGrad = (sqrt(pi)*sigma/(j*4*m*omega))*(ComputeUpsilon1 - ComputeUpsilon2);
    end
end;

if subComponent
    if size(meanVector,1) ==1,
        matGrad = matGrad*meanVector;
    else
        matGrad = (meanVector*matGrad).';
    end
end

g1(5) = sum(sum(matGrad.*covGrad));
g1 = real(g1);
% Gradient with respect to the "variance" of the RBF
g2(1) = 0; % Otherwise is counted twice, temporarly changed by MA
g2(2) = 0;

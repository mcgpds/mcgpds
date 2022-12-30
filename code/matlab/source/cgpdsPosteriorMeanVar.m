function [mu, varsigma] = cgpdsPosteriorMeanVar(model, X, varX)

% cgpdsPosteriorMeanVar Mean and variances of the posterior at points given by X.
%
%	Description:
%
%	[MU, SIGMA] = cgpdsPosteriorMeanVar(MODEL, X, VARX) returns the
%	unnormalized posterior mean and variance for a given set of points.
%	 Returns:
%	  MU - the mean of the posterior distribution.
%	  SIGMA - the variances of the posterior distributions.
%	 Arguments:
%	  MODEL - the model for which the posterior will be computed.
%	  X - variational mean in the latent space for which posterior is
%	   computed.
%	  VARX - variational variances in the latent space for which
%	   posterior is computed (assumed zero if not present).



% do prediction by replacing the variational distribution with a delta function  
%model.K_uf = kernCompute(model.kern, model.X_u, model.vardist.means);
%model.A = (1/model.beta)*model.K_uu + model.K_uf*model.K_uf';
%[model.Ainv, U] = pdinv(model.A);
%[mu1, varsigma1] = gpPosteriorMeanVar(model, vardistX.means);


% Find exactly the mean and the variances of the predictive distribution
% (which is not Gaussian, however its moments can be computed in closed-form)

if nargin < 3
  vardistX.covars = repmat(0.0, size(X, 1), size(X, 2));%zeros(size(X, 1), size(X, 2));
else
  vardistX.covars = varX;
end
vardistX.latentDimension = size(X, 2);
vardistX.numData = size(X, 1);
%model.vardist.covars = 0*model.vardist.covars; 
vardistX.means = X;
%model = vargplvmUpdateStats(model, model.X_u);


tempMatrix.C_v = model.invLm_v * model.Psi4 * model.invLmT_v;
tempMatrix.At_v = (1/model.beta) * eye(size(tempMatrix.C_v,1)) + tempMatrix.C_v; % At = beta^{-1} I + C
tempMatrix.Lat_v = jitChol(tempMatrix.At_v)';
tempMatrix.invLat_v = tempMatrix.Lat_v\eye(size(tempMatrix.Lat_v,1));  
tempMatrix.P1_v = tempMatrix.invLat_v * model.invLm_v; % M x M

Ainv_v = tempMatrix.P1_v' * tempMatrix.P1_v; % inv(model.K_uu/mode1.beta+model.Psi4) size: M*M

if ~isfield(model,'alpha_v')
    model.alpha_v = Ainv_v*model.Psi0'*model.m; % size: M*D
end

Psi0_star = kernVardistPsi1Compute(model.kern_v, vardistX, model.X_v);
meanh = Psi0_star*model.alpha_v;

mu = meanh;

Psi1_star = kernVardistPsi1Compute(model.kern_u, vardistX, model.X_u);
% Ainv_u = cell(model.d,model.J);
model.alpha_u = zeros(model.k,model.d,model.J);
meang = zeros(size(vardistX.means,1),model.d,model.J);


tempMatrix.C_u_temp = model.invLm_u * model.Psi5 * model.invLmT_u;  %M*M
for j = 1:model.J
    for d = 1:model.d
%         Ainv_u{d,j} = tempMatrix.P1_u{d,j}' * tempMatrix.P1_u{d,j};
%         model.alpha_u(:,d,j) = Ainv_u{d,j}*model.Psi1'*model.m(:,d); % size: M*1
        
        tempMatrix.C_u = model.W(d,j)^2 * tempMatrix.C_u_temp;
        tempMatrix.At_u = (1/model.beta) * eye(size(tempMatrix.C_u,1)) + tempMatrix.C_u; % At = beta^{-1} I + C
        tempMatrix.Lat_u = jitChol(tempMatrix.At_u)';%lower bound
        tempMatrix.invLat_u = tempMatrix.Lat_u\eye(size(tempMatrix.Lat_u,1)); 
        tempMatrix.P1_u = tempMatrix.invLat_u * model.invLm_u; % M x M
        
        Ainv_u = tempMatrix.P1_u' * tempMatrix.P1_u;
        model.alpha_u(:,d,j) = Ainv_u * model.Psi1'*model.m(:,d); % size: M*1
        meang(:,d,j) = model.W(d,j)*Psi1_star*model.alpha_u(:,d,j);%N^* * 1
    end
%     meang(:,:,j) = meang(:,:,j)*diag(model.W(:,j));
    mu = mu + meang(:,:,j);
end


varsigmah = zeros(size(vardistX.means,1),model.d);
varsigmag = zeros(size(vardistX.means,1),model.d,model.J);
varsigma = zeros(size(vardistX.means,1),model.d);

if nargout > 1  

   vard = vardistCreate(zeros(1,model.q), model.q, 'gaussian');%1*9 varidist
   Kinvk_v = (model.invK_vv - (1/model.beta)*Ainv_v);%model.invK_vv - inv(model.K_vv+model.beta*model.Psi4)
%    Kinvk_u = cell(model.d,model.J);
   vars_u = zeros(model.d,model.J);
   %
   for n=1:size(vardistX.means,1)
      %
      vard.means = vardistX.means(n,:);
      vard.covars = vardistX.covars(n,:);
      % compute psi0 term
      Psi2_star = kernVardistPsi0Compute(model.kern_v, vard);
      Psi3_star = kernVardistPsi0Compute(model.kern_u, vard);
      % compute psi2 term
      Psi4_star = kernVardistPsi2Compute(model.kern_v, vard, model.X_v);
      Psi5_star = kernVardistPsi2Compute(model.kern_u, vard, model.X_u);
      
      vars_v = Psi2_star - sum(sum(Kinvk_v.*Psi4_star));
      
      for d=1:model.d 
         %[model.alpha(:,j)'*(Psi2_star*model.alpha(:,j)), mu(i,j)^2]
         varsigmah(n,d) = model.alpha_v(:,d)'*(Psi4_star*model.alpha_v(:,d)) - meanh(n,d)^2;
         for j = 1:model.J
%              Kinvk_u{d,j} = (model.invK_uu - (1/model.beta)*Ainv_u{d,j});
%              vars_u(d,j) = Psi3_star - sum(sum(Kinvk_u{d,j}.*Psi5_star));
            % save memory
            tempMatrix.C_u = model.W(d,j)^2 * tempMatrix.C_u_temp;
            tempMatrix.At_u = (1/model.beta) * eye(size(tempMatrix.C_u,1)) + tempMatrix.C_u; % At = beta^{-1} I + C
            tempMatrix.Lat_u = jitChol(tempMatrix.At_u)';%lower bound
            tempMatrix.invLat_u = tempMatrix.Lat_u\eye(size(tempMatrix.Lat_u,1)); 
            tempMatrix.P1_u = tempMatrix.invLat_u * model.invLm_u; % M x M
        
            Ainv_u = tempMatrix.P1_u' * tempMatrix.P1_u;
            Kinvk_u = (model.invK_uu - (1/model.beta)*Ainv_u);
            vars_u(d,j) = Psi3_star - sum(sum(Kinvk_u.*Psi5_star));
            
             varsigmag(n,d,j) = model.alpha_u(:,d,j)'*(Psi5_star*model.alpha_u(:,d,j)) - meang(n,d,j)^2;
         end
         
      end
%       tempVarg = sum(varsigmag,3);
      varsigma(n,:) = varsigma(n,:) + varsigmah(n,:) + repmat(vars_v,1,model.d) ;
      for j = 1:model.J
          varsigma(n,:) = varsigma(n,:) + (model.W(:,j).^2)'.*(varsigmag(n,:,j) + vars_u(:,j)');
      end
       
   end
        if isfield(model, 'beta')
          varsigma = varsigma + (1/model.beta);
        end
end

clear tempMatrix;

function [ll, model] = vargplvmSeqDynLogLikelihood(model, modelAll, vardistx, y, samd)
% VARGPLVMSEQDYNLOGLIKELIHOOD Log-likelihood of a point for the GP-LVM.
% FORMAT
% DESC returns the log likelihood of a latent point and an observed
% data point for the posterior prediction of the GP-LVM model.
% ARG model : the model for which the point prediction will be
% made.
% ARG vardistx : the variational distribution over latent point for which the posterior distribution
% will be evaluated. It contains the mean and the diagonal covarriance 
% ARG y : the observed data point for which the posterior is evaluated
% ARG indexPresent: indicates which indices from the observed vector are present
%         (e.g. when when all indices are present, then y will d-dimensional     
%          and indexPresent = 1:D)          
%
% SEEALSO : vargplvmCreate, vargplvmOptimisePoint, vargplvmPointObjective
%
% COPYRIGHT : Michalis K. Titsias and Andreas Damianou, 2011

% VARGPLVM


% !!!!!! this function can become faster with precomputations stored in the
% structure model !!!!! 


% y is a new block/sequence 
Nstar = size(y,1);
N = model.N;

mask = sum(isnan(y),2); 
indexObservedData = find(mask==0)'; %%1*158
indexMissingData = setdiff(1:Nstar, indexObservedData); %%1*0

% Compute fully observed test data points and partially 
% observed data points 
yOb = y(indexObservedData, :); %%158*100
yMs = y(indexMissingData, :); %%158*0

% Indices of missing dimension in the Missingdata
indexMissing = []; %%0*0
indexPresent = [1:model.d]; %%1*100
if ~isempty(yMs)
   indexMissing = find(isnan(yMs(1,:)));
   indexPresent = setdiff(1:model.d, indexMissing);
   yMs = yMs(:,indexPresent);   
end
    
% normalize yOb and yMs exactly as model.m is normalized 
myOb = yOb; %%158*100
if ~isempty(yOb)
  myOb = yOb - repmat(model.bias,size(yOb,1),1); 
  myOb = myOb./repmat(model.scale,size(yOb,1),1); 
end


myMs = yMs; %%0*158
if ~isempty(yMs)
   myMs = yMs - repmat(model.bias(indexPresent),size(yMs,1),1);  
   myMs = myMs./repmat(model.scale(indexPresent),size(yMs,1),1);  
end
mOrig = model.m; %566*100

% re-order test data so that observed are first and then are the missing 
Order = [indexObservedData, indexMissingData]; %1*158
model.dynamics.t_star = model.dynamics.t_star(Order);
vardistx.means = vardistx.means(Order,:);
vardistx.covars = vardistx.covars(Order,:);
nObsData = size(indexObservedData,2);


% Form the modelTest for the new block which allows to compute the variational
% distribution (given the barmus and lambdas) for this new block. Notice that this distribution
% is independent from the other blocks in the training set. 
modelTest = model;
%modelTest.m = [myOb; myMs]; 
modelTest.dynamics.t = model.dynamics.t_star;
modelTest.dynamics.vardist = vardistx; 
modelTest.dynamics.N =  size(modelTest.dynamics.vardist.means, 1);
modelTest.N = size(modelTest.dynamics.vardist.means, 1);
modelTest.vardist.numData = modelTest.dynamics.N;
modelTest.vardist.nParams = 2*prod(size(modelTest.dynamics.vardist.means));
Kt = kernCompute(model.dynamics.kern, model.dynamics.t_star);
modelTest.dynamics.Kt = Kt;
modelTest.dynamics.At = (modelTest.alpha)^2*modelTest.dynamics.Kt + (1-modelTest.alpha)^2*modelTest.epsilon*eye(modelTest.N);
modelTest.X = modelTest.vardist.means;
modelTest.dynamics.X = modelTest.dynamics.vardist.means;
modelTest.dynamics.fixedKt = 1;
% This enables the computation of the variational means and covars
modelTest.dynamics.seq = []; 
modelTest = vargpTimeDynamicsUpdateStats(modelTest);

% The data in the test block/seq with missing values are processed to get their
% psi statisitcs. This is needed to form the LL2 term that is the part 
% of the bound corresponding to dimensions observed everywhere (in all
% training and test points)
if ~isempty(indexMissing)
    vardistMs = modelTest.vardist;
    vardistMs.means = modelTest.vardist.means(nObsData+1:end, :);
    vardistMs.covars = modelTest.vardist.covars(nObsData+1:end, :);
    vardistMs.nParams = 2*prod(size(vardistMs.means));
    vardistMs.numData = size(vardistMs.means,1);
    
    % Psi statistics for the data of the new block/sequence which have partially
    % observed dimensions
    missingPsi0 = kernVardistPsi1Compute(model.kern_v, vardistMs, model.X_v);
    missingPsi1 = kernVardistPsi1Compute(model.kern_u, vardistMs, model.X_u);
    
    missingPsi2 = kernVardistPsi0Compute(model.kern_v, vardistMs);
    missingPsi3 = kernVardistPsi0Compute(model.kern_u, vardistMs);
    
    missingPsi4 = kernVardistPsi2Compute(model.kern_v, vardistMs, model.X_v);
    missingPsi5 = kernVardistPsi2Compute(model.kern_u, vardistMs, model.X_u);
    
    model.Psi0 = [model.Psi0; missingPsi0];
    model.Psi1 = [model.Psi1; missingPsi1];  
    
    model.Psi2 = model.Psi2 + missingPsi2;
    model.Psi3 = model.Psi3 + missingPsi3;
    
    model.Psi4 = model.Psi4 + missingPsi4;
    model.Psi5 = model.Psi5 + missingPsi5;
    
    model.m = [mOrig(:,indexPresent); myOb(:, indexPresent); myMs];
    model.N =  N + Nstar;
    model.d = prod(size(indexPresent));
    
%     model.TrYY = sum(sum(model.m .* model.m));
%     model.C = model.invLm * model.Psi2 * model.invLmT;
%     model.TrC = sum(diag(model.C)); % Tr(C)
%     model.At = (1/model.beta) * eye(size(model.C,1)) + model.C; % At = beta^{-1} I + C
%     model.Lat = jitChol(model.At)';
%     model.invLat = model.Lat\eye(size(model.Lat,1));
%     model.invLatT = model.invLat';
%     model.logDetAt = 2*(sum(log(diag(model.Lat)))); % log |At|
%     model.P1 = model.invLat * model.invLm; % M x M
%     model.P = model.P1 * (model.Psi1' * model.m);
%     model.TrPP = sum(sum(model.P .* model.P));
end

% LL1 TERM
ll1 = 0;
if ~isempty(indexMissing)
%    dmis = prod(size(indexMissing));
%    
%    % Precompute again the parts that contain Y
%    TrYY = sum(sum(model.m(:,indexMissing) .* model.m(:,indexMissing)));
%    
%    ll1 = -0.5*(dmis*(-(model.N-model.k)*log(model.beta) ...
% 				  + model.logDetAt) ...
% 	      - (TrPP ...
% 	      - TrYY)*model.beta);
%    ll1 = ll1 - 0.5*model.beta*dmis*model.Psi0 + 0.5*dmis*model.beta*model.TrC;
%    ll1 = ll1-dmis*model.N/2*log(2*pi);

    samdMissing = intersect(samd,indexMissing);
    tempMatrix = calculateMatrix(model,samdMissing);
    dmis = prod(size(indexMissing));
    lendMissing = length(samdMissing);

    % Precompute again the parts that contain Y
    %    TrYY = sum(sum(model.m(:,indexMissing) .* model.m(:,indexMissing)));

    ll1 = -0.5*(lendMissing*(-(model.N-model.k)*log(model.beta) ...
        + tempMatrix.logDetAt_v) ...
        - (tempMatrix.TrPP_v ...
        - tempMatrix.TrYYdr)*model.beta);
    ll1 = ll1 - 0.5*model.beta*lendMissing*model.Psi2 + 0.5*lendMissing*model.beta*model.TrC_v;
    ll1 = ll1-lendMissing*model.N/2*log(2*pi);
    
    ll1 = ll1 - 0.5*(sum(sum(tempMatrix.logDetAt_u)) - lendMissing*model.J*(-model.k)*log(model.beta));
    ll1 = ll1 - 0.5*model.beta*sum(sum((model.W(samdMissing,:)).^2))*model.Psi3 + 0.5*model.beta*sum(sum(tempMatrix.TrC_u));
    
    ll1 = ll1 + 0.5*model.beta*tempMatrix.TrPP_u;
    
    ll1 = dmis/lendMissing*ll1;
end
% clear tempMatrix;

% The fully observed subset of data from the new block must be augmented with all previous
% training data to form the LL1 term in the whole bound 
if ~isempty(yOb)
   vardistOb = modelTest.vardist; 
   vardistOb.means = modelTest.vardist.means(1:nObsData, :);
   vardistOb.covars = modelTest.vardist.covars(1:nObsData, :);
   vardistOb.nParams = 2*prod(size(vardistOb.means));
   vardistOb.numData = size(vardistOb.means,1);
 
   % Psi statistics for the data of the new block/sequence which have a fully
   % observed features/dimensions
   obsPsi0 = kernVardistPsi1Compute(model.kern_v, vardistOb, model.X_v);
   obsPsi1 = kernVardistPsi1Compute(model.kern_u, vardistOb, model.X_u);
   
   obsPsi2 = kernVardistPsi0Compute(model.kern_v, vardistOb);
   obsPsi3 = kernVardistPsi0Compute(model.kern_u, vardistOb);
   
   obsPsi4 = kernVardistPsi2Compute(model.kern_v, vardistOb, model.X_v);
   obsPsi5 = kernVardistPsi2Compute(model.kern_u, vardistOb, model.X_u);

   model.N = model.N + size(yOb,1);
   model.m = [model.m; myOb];
   model.Psi0 = [model.Psi0; obsPsi0];
   model.Psi1 = [model.Psi1; obsPsi1];
   
   model.Psi2 = model.Psi2 + obsPsi2;
   model.Psi3 = model.Psi3 + obsPsi3;
   
   model.Psi4 = model.Psi4 + obsPsi4;
   model.Psi5 = model.Psi5 + obsPsi5;
   
   model.C_v = model.invLm_v * model.Psi4 * model.invLmT_v;
   model.TrC_v = sum(diag(model.C_v)); % Tr(C)
   model.At_v = (1/model.beta) * eye(size(model.C_v,1)) + model.C_v; % At = beta^{-1} I + C
   model.Lat_v = jitChol(model.At_v)';
   model.invLat_v = model.Lat_v\eye(size(model.Lat_v,1));
   model.invLatT_v = model.invLat_v';
   model.logDetAt_v = 2*(sum(log(diag(model.Lat_v)))); % log |At|
   model.P1_v = model.invLat_v * model.invLm_v; % M x M
   % once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
   model.P_v = model.P1_v * (model.Psi0' * model.m);
   model.TrPP_v = sum(sum(model.P_v .* model.P_v));
end 

% LL2 TERM 
ll2 = 0;
if ~isempty(indexPresent)   
    
    samdPresent = intersect(indexPresent,samd);
    partYMs = yMs(:, samdPresent);
    
    partYMs = partYMs - repmat(model.bias(:,samdPresent),size(partYMs,1),1);
    partYMs = partYMs./repmat(model.scale(:,samdPresent),size(partYMs,1),1);
    
    model.partm = [mOrig(:,samdPresent); myOb(:, samdPresent); partYMs]; %%%%%%partM(353+43)*lend
    tempMatrix = calculateMatrixwithPartM( model,samdPresent);
    lendPresent = length(samdPresent);
    dpre = length(indexPresent);
     
    ll2 = -0.5*(lendPresent*(-(model.N-model.k)*log(model.beta) ...
				  + tempMatrix.logDetAt_v) ...
	      - (tempMatrix.TrPP_v ...
	      - tempMatrix.TrYYdr)*model.beta);
    ll2 = ll2 - 0.5*model.beta*lendPresent*model.Psi2 + 0.5*lendPresent*model.beta*model.TrC_v;

    ll2 = ll2-lendPresent*model.N/2*log(2*pi);

    ll2 = ll2 - 0.5*(sum(sum(tempMatrix.logDetAt_u)) - lendPresent*model.J*(-model.k)*log(model.beta));
    ll2 = ll2 - 0.5*model.beta*sum(sum(model.W(samdPresent,:).^2))*model.Psi3 + 0.5*model.beta*sum(sum(tempMatrix.TrC_u));
    
    ll2 = ll2 + 0.5*model.beta*tempMatrix.TrPP_u;
    
    ll2 = dpre/lendPresent * ll2;
end
clear tempMatrix;

% KL TERM 

model.dynamics.t = [model.dynamics.t; model.dynamics.t_star];
% Augment the reparametrized variational parameters mubar and lambda
model.dynamics.vardist.means = [model.dynamics.vardist.means; vardistx.means];
model.dynamics.vardist.covars = [model.dynamics.vardist.covars; vardistx.covars]; 
model.dynamics.N = N + Nstar;
model.vardist.numData = model.dynamics.N;
Kt = zeros(N+Nstar,N+Nstar);
Kt(1:N,1:N) = model.dynamics.Kt; 
Kt(N+1:end, N+1:end) = modelTest.dynamics.Kt; 
model.dynamics.Kt = Kt;
model.dynamics.fixedKt = 1;
model.dynamics.seq = [model.dynamics.seq, (N+Nstar)];
A1 = zeros(N+Nstar,N+Nstar);
A1(1:N,1:N) = model.dynamics.At; 
A1(N+1:end, N+1:end) = modelTest.dynamics.At; 
model.dynamics.At = A1;

model.dynamics.vardist.Sq = cell(model.q,1);

for q=1:model.q
    LambdaH_q = model.dynamics.vardist.covars(:,q).^0.5;
    Bt_q = eye(model.N) + LambdaH_q*LambdaH_q'.*model.dynamics.At;
    
    % Invert Bt_q
    Lbt_q = jitChol(Bt_q)';
    G1 = Lbt_q \ diag(LambdaH_q);
    G = G1*model.dynamics.At;
    % Find Sq
    model.dynamics.vardist.Sq{q} = model.dynamics.At - G'*G;
end


Kt12 = zeros(N+Nstar,N+Nstar);
Kt12(1:N,1:N) = modelAll.dynamics.Kt; 
Kt12(N+1:end, N+1:end) = kernCompute(modelAll.dynamics.kern, model.dynamics.t_star); 
modelAll.dynamics.Kt = Kt12;

S1 = model.dynamics.vardist.Sq;
mu1 = model.dynamics.vardist.means;
alpha1 = modelAll.comp{1}.alpha;
alpha2 = modelAll.comp{2}.alpha;

mu12 = zeros(N+Nstar, model.q);
S12 = cell(model.q,1);
for q=1:model.q
    S12{q} = inv((1-alpha1)^2*inv(A1) + inv(Kt12));
    
    mu12(:,q) = S12{q}*((1-alpha1)*inv(A1) * mu1(:,q));
end

modelAll.dynamics.vardist.Sq = S12;
modelAll.dynamics.vardist.means = mu12;

% KLdiv = modelVarPriorBound(model);

KLdiv = 0;

for q=1:model.q
    if(det(A1)==0)
         %KLdiv = KLdiv - 1e5;
    else
         KLdiv = KLdiv + log(det(A1));
    end
     
     if(det(Kt12)==0)
         %KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv + log(det(Kt12));
     end
     if(det(S12{q})==0)
         %KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv - log(det(S12{q}));
     end
     if(det(S1{q})==0)
         %KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv - log(det(S1{q}));
     end
     
    KLdiv = KLdiv + ((1-alpha1)*mu12(:,q) - mu1(:,q))'*inv(A1)*((1-alpha1)*mu12(:,q) - mu1(:,q))...
        + trace(((1-alpha1)^2*inv(A1))*(S12{q}))...
        + trace(inv(Kt12)*(mu12(:,q)*mu12(:,q)'+ S12{q}))...
        + trace(inv(A1)*(S1{q}));
end

KLdiv = 0*0.5*KLdiv;

% sum all terms
ll = ll1 + ll2 - KLdiv; 

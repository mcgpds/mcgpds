function [g, model] = vargplvmSeqDynLogLikeGradient(model, modelAll, vardistx, y, samd)
% VARGPLVMSEQDYNLOGLIKEGRADIENT Log-likelihood gradient for of a point of the GP-LVM.
% FORMAT
% DESC returns the gradient of the log likelihood with respect to
% the latent position, where the log likelihood is conditioned on
% the training set. 
% ARG model : the model for which the gradient computation is being
% done.
% ARG x : the latent position where the gradient is being computed.
% ARG y : the position in data space for which the computation is
% being done.
% RETURN g : the gradient of the log likelihood, conditioned on the
% training data, with respect to the latent position.
%
% SEEALSO : vargplvmPointLogLikelihood, vargplvmOptimisePoint, vagplvmSequenceLogLikeGradient
%
% COPYRIGHT : Michalis K. Titsias and Andreas Damianou, 2011

% VARGPLVM


% y is a new block/sequence 
Nstar = size(y,1);
N = model.N;

mask = sum(isnan(y),2); 
indexObservedData = find(mask==0)'; 
indexMissingData = setdiff(1:Nstar, indexObservedData);

% Compute fully observed test data points and partially 
% observed data points 
yOb = y(indexObservedData, :);
yMs = y(indexMissingData, :);

% Indices of missing dimension in the Missingdata
indexMissing = [];
indexPresent = [1:model.d];
if ~isempty(yMs)
   indexMissing = find(isnan(yMs(1,:)));
   indexPresent = setdiff(1:model.d, indexMissing);
   yMs = yMs(:,indexPresent);   
end
    
% normalize yOb and yMs exactly as model.m is normalized 
myOb = yOb;
if ~isempty(yOb)
  myOb = yOb - repmat(model.bias,size(yOb,1),1); 
  myOb = myOb./repmat(model.scale,size(yOb,1),1); 
end
myMs = yMs;
if ~isempty(yMs)
   myMs = yMs - repmat(model.bias(indexPresent),size(yMs,1),1);  
   myMs = myMs./repmat(model.scale(indexPresent),size(yMs,1),1);  
end
mOrig = model.m;
%modelOrig = model;

% re-order test data so that observed are first and then are the missing 
Order = [indexObservedData, indexMissingData];
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
modelTest.dynamics.N = size(modelTest.dynamics.vardist.means, 1);
modelTest.N = size(modelTest.dynamics.vardist.means, 1);
modelTest.vardist.numData = modelTest.dynamics.N;
modelTest.vardist.nParams = 2*prod(size(modelTest.dynamics.vardist.means));
Kt = kernCompute(model.dynamics.kern, model.dynamics.t_star);
modelTest.dynamics.Kt = Kt;
modelTest.dynamics.fixedKt = 1;
modelTest.dynamics.At = (modelTest.alpha)^2*modelTest.dynamics.Kt + (1-modelTest.alpha)^2*modelTest.epsilon*eye(size(modelTest.dynamics.vardist.means,1));
    
% This enables the computation of the variational means and covars
modelTest.dynamics.seq = []; 
modelTest = vargpTimeDynamicsUpdateStats(modelTest);


% The fully observed subset of data from the new block must be augmented with all previous
% training data to form the LL1 term in the whole bound 
gVarmeansLik = zeros(1, nObsData*model.dynamics.q);
gVarcovsLik = zeros(1, nObsData*model.dynamics.q);
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
   
   model.N = N + size(yOb,1);
   model.m = [model.m; myOb];
   model.Psi0 = [model.Psi0; obsPsi0];
   model.Psi1 = [model.Psi1; obsPsi1];
   
   model.Psi2 = model.Psi2 + obsPsi2;
   model.Psi3 = model.Psi3 + obsPsi3;
   
   model.Psi4 = model.Psi4 + obsPsi4;
   model.Psi5 = model.Psi5 + obsPsi5;
   
   model.C_v = model.invLm_v * model.Psi4 * model.invLmT_v;
   model.At_v = (1/model.beta) * eye(size(model.C_v,1)) + model.C_v; % At = beta^{-1} I + C
   model.Lat_v = jitChol(model.At_v)';
   model.invLat_v = model.Lat_v\eye(size(model.Lat_v,1));
   model.P1_v = model.invLat_v * model.invLm_v; % M x M
   P1TP1_v = (model.P1_v' * model.P1_v);
   model.P_v = model.P1_v * (model.Psi0' * model.m);
   model.B_v = model.P1_v' * model.P_v;
   Tb_v = (1/model.beta) * model.d * (model.P1_v' * model.P1_v);
   Tb_v = Tb_v + (model.B_v * model.B_v');
   model.T1_v = model.d * model.invK_vv - Tb_v;
   
   gPsi0_full = model.beta*(P1TP1_v*model.Psi0'*model.m*model.m');
   gPsi0 = gPsi0_full(:,N+1:N+Nstar);
   gPsi2 = -0.5 * model.beta * model.d;
   gPsi4 = (model.beta/2) * model.T1_v;
  
   samdPresent = intersect(samd,indexPresent);
   lendPresent = length(samdPresent);
   dpre = length(indexPresent);
   
   partYMs = yMs(:, samdPresent);
   partYMs = partYMs - repmat(model.bias(:,samdPresent),size(partYMs,1),1);
   partYMs = partYMs./repmat(model.scale(:,samdPresent),size(partYMs,1),1);
   
   model.partm = [mOrig(:,samdPresent); myOb(:, samdPresent); partYMs];
   
   gPsi1_full = zeros(N+Nstar,model.k);
   gPsi3 = 0;
   gPsi5 = zeros(model.k,model.k);
   
   for d = 1:length(samdPresent)
       index = samdPresent(d);
       for j = 1:model.J
           %         gPsi1_full = gPsi1_full + model.beta * model.W(index,j)^2 * (P1TP1_u{d,j}*model.Psi1'*model.partm(:,d)*model.partm(:,d)');%N*M
           gPsi1_full = gPsi1_full + model.beta*model.W(index,j)^2*model.partm(:,d)*model.partm(:,d)'*model.Psi1...
               *inv(model.W(index,j)^2*model.Psi5+model.K_uu/model.beta);
           
           gPsi3 = gPsi3 - 0.5 * model.beta * model.W(index,j)^2;
           
           gPsi5 = gPsi5 + 0.5*model.beta*model.W(index,j)^2*model.invK_uu...
               -0.5*model.beta*model.W(index,j)^4*inv(model.W(index,j)^2*model.Psi5+model.K_uu/model.beta)*...
               model.Psi1'*model.partm(:,d)*model.partm(:,d)'*model.Psi1*inv(model.W(index,j)^2*model.Psi5+model.K_uu/model.beta)...
               -0.5*model.W(index,j)^2*inv(model.W(index,j)^2*model.Psi5+model.K_uu/model.beta);
       end
   end
   gPsi1 = gPsi1_full(N+1:N+Nstar,:)';

   gPsi1 = dpre/lendPresent * gPsi1;
   gPsi3 = dpre/lendPresent * gPsi3;
   gPsi5 = dpre/lendPresent * gPsi5;
   
   [gKern0, gVarmeans0, gVarcovs0, gInd0] = kernVardistPsi1Gradient(model.kern_v, modelTest.vardist, model.X_v, gPsi0');
   [gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model.kern_u, modelTest.vardist, model.X_u, gPsi1');
   
   [gKern2, gVarmeans2, gVarcovs2] = kernVardistPsi0Gradient(model.kern_v, modelTest.vardist, gPsi2);
   [gKern3, gVarmeans3, gVarcovs3] = kernVardistPsi0Gradient(model.kern_u, modelTest.vardist, gPsi3);
   
   [gKern4, gVarmeans4, gVarcovs4, gInd4] = kernVardistPsi2Gradient(model.kern_v, modelTest.vardist, model.X_v, gPsi4);
   [gKern5, gVarmeans5, gVarcovs5, gInd5] = kernVardistPsi2Gradient(model.kern_u, modelTest.vardist, model.X_u, gPsi5);
   
   gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2 + gVarmeans3 + gVarmeans4 + gVarmeans5;
   gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
end 

model.dynamics.t = [model.dynamics.t; model.dynamics.t_star];
% Augment the reparametrized variational parameters mubar and lambda
model.dynamics.vardist.means = [model.dynamics.vardist.means; vardistx.means];
model.dynamics.vardist.covars = [model.dynamics.vardist.covars; vardistx.covars]; 
model.dynamics.N = N + Nstar;
model.vardist.numData = model.dynamics.N;

% GRADIENT FOR THE LL1 TERM
gPointDyn1 = zeros(Nstar, model.dynamics.q*2);
if ~isempty(indexMissing)
   % means
%    gPointDyn1(:,1:model.dynamics.q) = modelTest.dynamics.Kt(1:nObsData,:)'*reshape(gVarmeansLik, nObsData, model.q);
    gPointDyn1(:,1:model.dynamics.q) = gVarmeansLik - inv(modelTest.dynamics.At)*(modelTest.dynamics.vardist.means - (1-modelTest.alpha)*modelAll.dynamics.vardist.means);
   % covars
   gVarcovsLik = reshape(gVarcovsLik, nObsData, model.q);
   gcovTmp = zeros(Nstar,model.dynamics.q);
   for q=1:model.dynamics.q
       LambdaH_q = modelTest.dynamics.vardist.covars(:,q).^0.5;
       Bt_q = eye(Nstar) + LambdaH_q*LambdaH_q'.*modelTest.dynamics.At;
       % Invert Bt_q
       Lbt_q = jitChol(Bt_q)';
       G1 = Lbt_q \ diag(LambdaH_q);
       G = G1*modelTest.dynamics.At;
       % Find Sq
       Sq = modelTest.dynamics.At - G'*G;
       Sq = - (Sq .* Sq);
       % only the cross matrix is needed as in barmu case
       gcovTmp(:,q) = Sq(1:nObsData,:)'*gVarcovsLik(:,q);
   end
   gPointDyn1(:,(model.dynamics.q+1):(model.dynamics.q*2)) = gcovTmp.*modelTest.dynamics.vardist.covars;
%   
end

% GRADIENT FOR THE LL2 TERM PLUS KL TERM
%
% % The data in the test block/seq with missing values are processed to get their
% % psi statisitcs. This is needed to form the LL2 term that is the part 
% % of the bound corresponding to dimensions observed everywhere (in all
% % training and test points)
% vardistMs = modelTest.vardist; 
% vardistMs.means = modelTest.vardist.means(nObsData+1:end, :);
% vardistMs.covars = modelTest.vardist.covars(nObsData+1:end, :);
% vardistMs.nParams = 2*prod(size(vardistMs.means));
% vardistMs.numData = size(vardistMs.means,1);
% % Psi statistics for the data of the new block/sequence which have partially
% % observed dimensions
% missingPsi0 = kernVardistPsi0Compute(model.kern, vardistMs);
% missingPsi1 = kernVardistPsi1Compute(model.kern, vardistMs, model.X_u);
% missingPsi2 = kernVardistPsi2Compute(model.kern, vardistMs, model.X_u);  
% model.m = [mOrig(:,indexPresent); myOb(:, indexPresent); myMs];  
% model.N = N + Nstar;
% model.d = prod(size(indexPresent));
% model.dynamics.nParams = model.dynamics.nParams + 2*prod(size(vardistx.means));
% model.nParams = model.nParams + 2*prod(size(vardistx.means));
% model.Psi1 = [model.Psi1; missingPsi1]; 
% model.Psi2 = model.Psi2 + missingPsi2;
% model.Psi0 = model.Psi0 + missingPsi0;
% model.C = model.invLm * model.Psi2 * model.invLmT;
% model.At = (1/model.beta) * eye(size(model.C,1)) + model.C; % At = beta^{-1} I + C
% model.Lat = jitChol(model.At)';
% model.invLat = model.Lat\eye(size(model.Lat,1));  
% model.P1 = model.invLat * model.invLm; % M x M
% P1TP1 = (model.P1' * model.P1);
% model.P = model.P1 * (model.Psi1' * model.m);
% model.B = model.P1' * model.P;
% Tb = (1/model.beta) * model.d * (model.P1' * model.P1);
%      Tb = Tb + (model.B * model.B');
% model.T1 = model.d * model.invK_uu - Tb;
%      
% % Precompuations for the gradients
% gPsi1 = model.beta*(P1TP1*model.Psi1'*model.m*[myOb(:,indexPresent); myMs]');
% gPsi2 = (model.beta/2) * model.T1;
% gPsi0 = -0.5 * model.beta * model.d;
% [gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model.kern, modelTest.vardist, model.X_u, gPsi1');
% [gKern2, gVarmeans2, gVarcovs2, gInd2] = kernVardistPsi2Gradient(model.kern, modelTest.vardist, model.X_u, gPsi2);
% [gKern0, gVarmeans0, gVarcovs0] = kernVardistPsi0Gradient(model.kern, modelTest.vardist, gPsi0);
% gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2;
% gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2;      
% %

gPointDyn2 = zeros(Nstar, model.dynamics.q*2); 
% means
A1 = zeros(N+Nstar,N+Nstar);
A1(1:N,1:N) = model.dynamics.At; 
A1(N+1:end, N+1:end) = modelTest.dynamics.At; 

mu1 = model.dynamics.vardist.means;

alpha1 = modelAll.comp{1}.alpha;

Kt12 = zeros(N+Nstar,N+Nstar);
Kt12(1:N,1:N) = modelAll.dynamics.Kt; 
Kt12(N+1:end, N+1:end) = kernCompute(modelAll.dynamics.kern, model.dynamics.t_star); 

mu12 = zeros(N+Nstar, model.q);
S12 = cell(model.q,1);
for q=1:model.q
    S12{q} = inv((1-alpha1)^2*inv(A1)...
        + inv(Kt12));
    
    mu12(:,q) = S12{q}*((1-alpha1)*inv(A1)...
        *mu1(:,q));
end

modelAll.dynamics.vardist.Sq = S12;
modelAll.dynamics.vardist.means = mu12;

gVarmeansLik = reshape(gVarmeansLik, Nstar, model.dynamics.q);
%gPointDyn2(:,1:model.dynamics.q) = gVarmeansLik - inv(modelTest.dynamics.At)*(modelTest.dynamics.vardist.means - (1-modelTest.alpha)*modelAll.dynamics.vardist.means(N+1:end,:));
gVarmeansKL = inv(A1)*(mu1 - (1-alpha1)*mu12);

gPointDyn2(:,1:model.dynamics.q) = (gVarmeansLik - 0*gVarmeansKL(N+1:end,:));
% covars
gVarcovsLik = reshape(gVarcovsLik, Nstar, model.q);
gcovTmp = zeros(Nstar,model.dynamics.q);
for q=1:model.dynamics.q
    LambdaH_q = modelTest.dynamics.vardist.covars(:,q).^0.5;
    Bt_q = eye(Nstar) + LambdaH_q*LambdaH_q'.*modelTest.dynamics.At;
    % Invert Bt_q
    Lbt_q = jitChol(Bt_q)';
    G1 = Lbt_q \ diag(LambdaH_q);
    G = G1*modelTest.dynamics.At;
    % Find Sq
    Sq = modelTest.dynamics.At - G'*G;
    gcovTmp(:,q) = - (Sq .* Sq) * (gVarcovsLik(:,q) + 0.5*modelTest.dynamics.vardist.covars(:,q));
end
gPointDyn2(:,(model.dynamics.q+1):(model.dynamics.q*2)) = 0*gcovTmp.*modelTest.dynamics.vardist.covars; 

gPointDyn = gPointDyn1 + gPointDyn2;

% applying the inverse Order to give the gradeitn in the 
%orginal order of the data
T = eye(Nstar);
T = T(Order,:);
% inverse permutation
T = T';
% there must be a better way to do this
InvOrder = zeros(1,Nstar);
for i=1:Nstar
    InvOrder(i) = find(T(i,:)==1); 
end
gPointDyn = gPointDyn(InvOrder, :);

g = gPointDyn(:)';

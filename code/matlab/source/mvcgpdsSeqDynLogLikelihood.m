function [ll, model] = mvcgpdsSeqDynLogLikelihood(model, modelAll, vardistx, y, samd)
% mvcgpdsSeqDynLogLikelihood Log-likelihood of the mvcgpds for test data.

% y is a new block/sequence 
Nstar = size(y,1);

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

   
   modelTest.m = myOb;
   modelTest.Psi0 = obsPsi0;
   modelTest.Psi1 = obsPsi1;
   
   modelTest.Psi2 =  obsPsi2;
   modelTest.Psi3 =  obsPsi3;
   
   modelTest.Psi4 = obsPsi4;
   modelTest.Psi5 = obsPsi5;
   
   model = modelTest;
   
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

% LL TERM 
ll = 0;
if ~isempty(indexPresent)
    samdPresent = intersect(indexPresent,samd);
    
    model.m =  myOb(:, samdPresent);
    tempMatrix = calculateMatrix( model,samdPresent);
    lendPresent = length(samdPresent);
    dpre = length(indexPresent);
    
    ll = -0.5*(lendPresent*(-(model.N-model.k)*log(model.beta) ...
        + tempMatrix.logDetAt_v ) ...
        - (tempMatrix.TrPP_v- tempMatrix.TrYYdr)*model.beta);
    %if strcmp(model.approx, 'dtcvar')
    ll = ll - 0.5*model.beta*lendPresent*model.Psi2 + 0.5*lendPresent*model.beta*tempMatrix.TrC_v;
    
    ll = ll - 0.5*(sum(sum(tempMatrix.logDetAt_u)) - lendPresent*model.J*(-model.k)*log(model.beta));
    %if strcmp(model.approx, 'dtcvar')
    ll = ll - 0.5*model.beta*sum(sum((model.W(samd,:)).^2))*model.Psi3 + 0.5*model.beta*sum(sum(tempMatrix.TrC_u));
    
    ll = ll + 0.5*model.beta*tempMatrix.TrPP_u;
   
    ll = ll-lendPresent*model.N/2*log(2*pi);
end
ll = dpre/lendPresent*ll;
clear tempMatrix;

% KL TERM 
model.dynamics.t =  model.dynamics.t_star;
% Augment the reparametrized variational parameters mubar and lambda
model.dynamics.vardist.means = vardistx.means;
model.dynamics.vardist.covars =  vardistx.covars; 
model.dynamics.N = Nstar;
model.vardist.numData = model.dynamics.N;
Kt = modelTest.dynamics.Kt; 
model.dynamics.fixedKt = 1;
model.dynamics.seq = [];
A1 = modelTest.dynamics.At; 
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

Kt12 = kernCompute(modelAll.dynamics.kern, model.dynamics.t_star); 
modelAll.dynamics.Kt = Kt12;

S1 = model.dynamics.vardist.Sq;
mu1 = model.dynamics.vardist.means;
alpha1 = model.alpha;

mu12 = zeros(Nstar, model.q);
S12 = cell(model.q,1);
for q=1:model.q
    S12{q} = inv((1-alpha1)^2*inv(A1) + inv(Kt12));
    mu12(:,q) = S12{q}*((1-alpha1)*inv(A1)*mu1(:,q));
end

modelAll.dynamics.vardist.Sq = S12;
modelAll.dynamics.vardist.means = mu12;

KLdiv = 0;

for q=1:model.q
    if(det(A1)==0)
         KLdiv = KLdiv - 1e5;
    else
         KLdiv = KLdiv + log(det(A1));
    end
     
     if(det(Kt12)==0)
         KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv + log(det(Kt12));
     end
     if(det(S12{q})==0)
         KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv - log(det(S12{q}));
     end
     if(det(S1{q})==0)
         KLdiv = KLdiv - 1e5;
     else
         KLdiv = KLdiv - log(det(S1{q}));
     end
     KLdiv = KLdiv + ((1-alpha1)*mu12(:,q) - mu1(:,q))'*inv(A1)*((1-alpha1)*mu12(:,q) - mu1(:,q))...
        + trace(((1-alpha1)^2*inv(A1))*(S12{q}))...
        + trace(inv(Kt12)*(mu12(:,q)*mu12(:,q)'+ S12{q}))...
        + trace(inv(A1)*(S1{q}));
end

KLdiv = 0.5*KLdiv;

if isfield(modelAll,'fixll') && modelAll.fixll == 1
    ll = 0*ll;
end
% sum all terms
ll =  ll - KLdiv; 

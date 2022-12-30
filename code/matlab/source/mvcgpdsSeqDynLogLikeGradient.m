function [g, model] = mvcgpdsSeqDynLogLikeGradient(model, modelAll, vardistx, y, samd)
% mvcgpdsSeqDynLogLikeGradient Gradients of the mvcgpds for test data.

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
model.y = y;
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
   
   modelTest.N = size(yOb,1);
   modelTest.m = myOb;
   modelTest.Psi0 = obsPsi0;
   modelTest.Psi1 = obsPsi1;
   
   modelTest.Psi2 = obsPsi2;
   modelTest.Psi3 = obsPsi3;
   
   modelTest.Psi4 = obsPsi4;
   modelTest.Psi5 = obsPsi5;
   
   model = modelTest;
   
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
   
   gPsi0 = model.beta * model.m * model.B_v'; %Npart*M
   gPsi0 = gPsi0'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...M*N
   gPsi2 = -0.5 * model.beta * model.d; %1*1
   gPsi4 = (model.beta/2) * model.T1_v; %30*30
   
   samdPresent = intersect(samd,indexPresent);
   lendPresent = length(samdPresent);
   dpre = length(indexPresent);
   tempMatrix = calculateMatrix( model,samdPresent);
   model.m = myOb(:, samdPresent);
   
   gPsi1_temp = zeros(Nstar,model.k);
   gPsi1 = gPsi1_temp';
   gPsi3 = 0;
   gPsi5 = zeros(model.k,model.k);
   
   if ~(isfield(model,'fixU') && model.fixU == 1)
       for d = 1:lendPresent
           dd = samdPresent(d);
           for j = 1:model.J
               gPsi1_temp = gPsi1_temp + model.beta * model.W(dd,j)^2 * model.m(:,dd) * tempMatrix.B_u{d,j}';%N*M
               gPsi3 = gPsi3 - 0.5 * model.beta * model.W(dd,j)^2;
           end
       end
       gPsi5 = (model.beta/2) * tempMatrix.T1_u; %30*30
       
       % gPsi1 = model.beta * model.W^2 * model.m * model.B_u'; %30*359
       gPsi1 = dpre/lendPresent*gPsi1_temp'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...
       gPsi3 = dpre/lendPresent*gPsi3;
       gPsi5 = dpre/lendPresent * gPsi5;
   end
   
   
   [gKern0, gVarmeans0, gVarcovs0, gInd0] = kernVardistPsi1Gradient(model.kern_v, modelTest.vardist, model.X_v, gPsi0');
   [gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model.kern_u, modelTest.vardist, model.X_u, gPsi1');
   
   [gKern2, gVarmeans2, gVarcovs2] = kernVardistPsi0Gradient(model.kern_v, modelTest.vardist, gPsi2);
   [gKern3, gVarmeans3, gVarcovs3] = kernVardistPsi0Gradient(model.kern_u, modelTest.vardist, gPsi3);
   
   [gKern4, gVarmeans4, gVarcovs4, gInd4] = kernVardistPsi2Gradient(model.kern_v, modelTest.vardist, model.X_v, gPsi4);
   [gKern5, gVarmeans5, gVarcovs5, gInd5] = kernVardistPsi2Gradient(model.kern_u, modelTest.vardist, model.X_u, gPsi5);
   
   gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2 + gVarmeans3 + gVarmeans4 + gVarmeans5;
   gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
end 

model.dynamics.t = model.dynamics.t_star;
% Augment the reparametrized variational parameters mubar and lambda
model.dynamics.vardist.means =  vardistx.means;
model.dynamics.vardist.covars = vardistx.covars; 
model.dynamics.N = Nstar;
model.vardist.numData = model.dynamics.N;

gPointDyn2 = zeros(Nstar, model.dynamics.q*2); 
% means
A1 = modelTest.dynamics.At; 

mu1 = model.dynamics.vardist.means;

alpha1 = modelAll.comp{1}.alpha;

Kt12 = kernCompute(modelAll.dynamics.kern, model.dynamics.t_star); 

mu12 = zeros(Nstar, model.q);
S12 = cell(model.q,1);
for q=1:model.q
    S12{q} = inv((1-alpha1)^2*inv(A1) + inv(Kt12));
    mu12(:,q) = S12{q}*((1-alpha1)*inv(A1)*mu1(:,q));
end

modelAll.dynamics.vardist.Sq = S12;
modelAll.dynamics.vardist.means = mu12;

gVarmeansLik = reshape(gVarmeansLik, Nstar, model.dynamics.q);
%gPointDyn2(:,1:model.dynamics.q) = gVarmeansLik - inv(modelTest.dynamics.At)*(modelTest.dynamics.vardist.means - (1-modelTest.alpha)*modelAll.dynamics.vardist.means(N+1:end,:));
gVarmeansKL = inv(A1)*(mu1 - (1-alpha1)*mu12);

if isfield(modelAll,'fixll') && modelAll.fixll == 1
    gVarmeansLik = 0*gVarmeansLik;
end
gPointDyn2(:,1:model.dynamics.q) = (gVarmeansLik - gVarmeansKL);
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

if isfield(modelAll,'fixcovars') && modelAll.fixcovars == 1
    gcovTmp = 0*gcovTmp;
end
% gPointDyn2(:,(model.dynamics.q+1):(model.dynamics.q*2)) = gcovTmp.*modelTest.dynamics.vardist.covars; 

gPointDyn = gPointDyn2;

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

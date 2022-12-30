function model = mvcgpdsExpandParam(model, params, samd)

% mvcgpdsExpandParam Expand a parameter vector into a multi-view cgpds model.
% FORMAT
% DESC takes a multi-view cgpds structure and a vector of parameters, and
% fills the structure with the given parameters. Also performs any
% necessary precomputation for likelihood and gradient
% computations, so can be computationally intensive to call.
% Parallelism can be applied w.r.t the models, so that some speed-up is achieved on
% multi-core machines.

% Parameters must be passed as a vector in the following order (left to right) 
% - parameter{size} -
% vardistParams{model.vardist.nParams} % mu, S
%       OR
% [dynamicsVardistParams{dynamics.vardist.nParams} dynamics.kernParams{dynamics.kern.nParams}] % mu_bar, lambda
%    % Followed by:
% private params of 1st model {model.comp{1}.nPrivateParams}
% private params of 2nd model {model.comp{2}.nPrivateParams}
%           ...
% private params of i-th model {model.comp{i}.nPrivateParams}
%
% ARG model : the svargplvm model to update with parameters
% ARG params : parameter vector
% RETURN model : model with updated parameters
%
% COPYRIGHT : Andreas C. Damianou, 2011



try
    p = gcp('nocreate');
    pool_open = p.NumWorkers;
%     pool_open = matlabpool('size')>0;
catch e
    pool_open = 0;
end

if pool_open && (isfield(model,'parallel') && model.parallel)
    model = mvcgpdsExpandParamPar(model, params);
else
    model = mvcgpdsExpandParamOrig(model, params, samd);
end
%!!! Since all dynamical models have the same "dynamics" structure, we can
%avoid the expandParam for ALL the dynamics structures. In fact, we should
%not store all these dynamics structures at all.

function model = mvcgpdsExpandParamOrig(model, params ,samd)

%startVal = 1;
%endVal = model.N*model.q;
%paramsX = params(startVal:endVal);
% model.barmu = reshape(params(startVal:endVal), model.N, model.q);

startVal=1;

if isfield(model, 'dynamics') & ~isempty(model.dynamics)
    endVal = model.dynamics.nParams;
else
    endVal = model.vardist.nParams;
end
sharedParams = params(startVal:endVal);

model.dynamics = modelExpandParam(model.dynamics, sharedParams, samd);

for i = 1:model.numModels
    samd = [1:model.comp{i}.d];
    % model.comp{i}.X = model.X;
    startVal = endVal+1;
    endVal = startVal + model.comp{i}.nPrivateParams-1; 
    %params_i = [sharedParams params(startVal:endVal)];
    params_i = params(startVal:endVal);
    model.comp{i} = cgpdsExpandParam(model.comp{i}, params_i, samd);
    
    model.comp{i}.dynamics.At = (model.comp{i}.alpha)^2*model.comp{i}.dynamics.Kt + (1-model.comp{i}.alpha)^2*model.comp{i}.epsilon*eye(model.comp{i}.N);
    
    model.comp{i}.dynamics.vardist.Sq = cell(model.comp{i}.q,1);
    
    for q=1:model.comp{i}.q
        LambdaH_q = model.comp{i}.dynamics.vardist.covars(:,q).^0.5;
        Bt_q = eye(model.comp{i}.N) + LambdaH_q*LambdaH_q'.*model.comp{i}.dynamics.At;
        
        % Invert Bt_q
        Lbt_q = jitChol(Bt_q)';
        G1 = Lbt_q \ diag(LambdaH_q);
        G = G1*model.comp{i}.dynamics.At;
        % Find Sq
        model.comp{i}.dynamics.vardist.Sq{q} = model.comp{i}.dynamics.At - G'*G;
    end
    % For compatibility ...
    if strcmp(model.comp{i}.kern_v.type, 'rbfardjit')
        model.comp{i}.kern_v.comp{1}.inputScales = model.comp{i}.kern_v.inputScales;
    end
    if strcmp(model.comp{i}.kern_u.type, 'rbfardjit')
        model.comp{i}.kern_u.comp{1}.inputScales = model.comp{i}.kern_u.inputScales;
    end
    %[par, names] = vargplvmExtractParam(model.comp{i}); %%%%5
    %for j=5:-1:1 %%
    %params_i(end-j) %%%
    %end%%
    %model.comp{i}.kern.comp{1}.inputScales
end

%optimize share variational parameters means and covariance matrices after
%optimizing private variational parameters means and covariance matrices
model.dynamics.vardist.Sq = cell(model.q,1);
for q=1:model.q
    model.dynamics.vardist.Sq{q} = inv((1-model.comp{1}.alpha)^2*inv(model.comp{1}.dynamics.At)...
        + (1-model.comp{2}.alpha)^2*inv(model.comp{2}.dynamics.At)...
        + inv(model.dynamics.Kt));
    
    model.dynamics.vardist.means(:,q) = model.dynamics.vardist.Sq{q}*((1-model.comp{1}.alpha)*inv(model.comp{1}.dynamics.At)...
        *model.comp{1}.dynamics.vardist.means(:,q) + (1-model.comp{2}.alpha)*inv(model.comp{2}.dynamics.At)*model.comp{2}.dynamics.vardist.means(:,q));
end

% with cgpdsExpandParam for the submodels, and given that the dynamics
% parameters (if there are dynamics) are shared, barmu has been turned into
% mu and this is the same for all submodels. The same holds for the
% variational distribution.
%model.X = model.comp{1}.X;

% if isfield(model, 'dynamics') & ~isempty(model.dynamics)
%     model.dynamics = model.comp{1}.dynamics; % = model.comp{i}.dynamics for all submodels m
% end
% 
% model.vardist = model.comp{1}.vardist;



function model = mvcgpdsExpandParamPar(model, params ,samd)
%startVal = 1;
%endVal = model.N*model.q;
%paramsX = params(startVal:endVal);
% model.barmu = reshape(params(startVal:endVal), model.N, model.q);
startVal=1;

if isfield(model, 'dynamics') & ~isempty(model.dynamics)
    endVal = model.dynamics.nParams;
else
    endVal = model.vardist.nParams;
end
sharedParams = params(startVal:endVal);

model.dynamics = modelExpandParam(model.dynamics, sharedParams, samd);

startVal{1} = endValS+1;
endVal{1} = startVal{1} + model.comp{1}.nPrivateParams-1;
for i = 2:model.numModels
    % model.comp{i}.X = model.X;
    startVal{i} = endVal{i-1}+1;
    endVal{i} = startVal{i} + model.comp{i}.nPrivateParams-1; 
end

modelTempComp = model.comp;
parfor i = 1:model.numModels
    % model.comp{i}.X = model.X;
   % startVal = endVal+1;
   % endVal = startVal + model.comp{i}.nPrivateParams-1; 
   samd = [1:modelTempComp{i}.d];
    params_i = [sharedParams params(startVal{i}:endVal{i})];
    modelTempComp{i} = cgpdsExpandParam(modelTempComp{i}, params_i, samd);
    if strcmp(modelTempComp{i}.kern.type, 'rbfardjit') || ~iscell(modelTempComp{i}.kern)
        modelTempComp{i}.kern.comp{1}.inputScales = modelTempComp{i}.kern.inputScales;
    end
    
    modelTempComp{i}.dynamics.At = (modelTempComp{i}.alpha)^2*modelTempComp{i}.dynamics.Kt + (1-modelTempComp{i}.alpha)^2*modelTempComp{i}.epsilon*eye(modelTempComp{i}.N);
    
    modelTempComp{i}.dynamics.vardist.Sq = cell(modelTempComp{i}.q,1);
    
    for q=1:modelTempComp{i}.q
        LambdaH_q = modelTempComp{i}.dynamics.vardist.covars(:,q).^0.5;
        Bt_q = eye(modelTempComp{i}.N) + LambdaH_q*LambdaH_q'.*modelTempComp{i}.dynamics.At;
        
        % Invert Bt_q
        Lbt_q = jitChol(Bt_q)';
        G1 = Lbt_q \ diag(LambdaH_q);
        G = G1*modelTempComp{i}.dynamics.At;
        % Find Sq
        modelTempComp{i}.dynamics.vardist.Sq{q} = modelTempComp{i}.dynamics.At - G'*G;
    end
    % For compatibility ...
    if strcmp(modelTempComp{i}.kern_v.type, 'rbfardjit')
        modelTempComp{i}.kern_v.comp{1}.inputScales = modelTempComp{i}.kern_v.inputScales;
    end
    if strcmp(modelTempComp{i}.kern_u.type, 'rbfardjit')
        modelTempComp{i}.kern_u.comp{1}.inputScales = modelTempComp{i}.kern_u.inputScales;
    end
    %[par, names] = vargplvmExtractParam(model.comp{i}); %%%%5
    %for j=5:-1:1 %%
    %params_i(end-j) %%%
    %end%%
    %model.comp{i}.kern.comp{1}.inputScales
end
model.comp = modelTempComp;

model.dynamics.vardist.Sq = cell(model.q,1);
for q=1:model.q
    model.dynamics.vardist.Sq{q} = inv((1-model.comp{1}.alpha)^2*inv(model.comp{1}.dynamics.At)...
        + (1-model.comp{2}.alpha)^2*inv(model.comp{2}.dynamics.At)...
        + inv(model.dynamics.Kt));
    
    model.dynamics.vardist.means(:,q) = model.dynamics.vardist.Sq{q}*((1-model.comp{1}.alpha)*inv(model.comp{1}.dynamics.At)...
        *model.comp{1}.dynamics.vardist.means(:,q) + (1-model.comp{2}.alpha)*inv(model.comp{2}.dynamics.At)*model.comp{2}.dynamics.vardist.means(:,q));
end


% with vargplvmExpandParam for the submodels, and given that the dynamics
% parameters (if there are dynamics) are shared, barmu has been turned into
% mu and this is the same for all submodels. The same holds for the
% variational distribution.
% model.X = model.comp{1}.X;
% 
% if isfield(model, 'dynamics') & ~isempty(model.dynamics)
%     model.dynamics = model.comp{1}.dynamics; % = model.comp{i}.dynamics for all submodels m
% end
% 
% model.vardist = model.comp{1}.vardist;





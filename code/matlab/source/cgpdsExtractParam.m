function [params, names] = cgpdsExtractParam(model, samd)

% cgpdsExtractParam Extract a parameter vector from a cgpds model.
%
%	Description:
%
%	PARAMS = cgpdsExtractParam(MODEL) extracts a parameter vector
%	from a given cgpds structure.
%	 Returns:
%	  PARAMS - the parameter vector extracted from the model.
%	 Arguments:
%	  MODEL - the model from which parameters are to be extracted.
%	DESC does the same as above, but also returns parameter names.
%	ARG model : the model structure containing the information about
%	the model.
%	RETURN params : a vector of parameters from the model.
%	RETURN names : cell array of parameter names.
%	

%%% Parameters must be returned as a vector in the following order (left to right) 
% - parameter{size} -
% vardistParams{model.vardist.nParams} % mu, S
%       OR
% [dynamicsVardistParams{dynamics.vardist.nParams} dynamics.kernParams{dynamics.kern.nParams}] % mu_bar, lambda
% inducingInputs{model.q*model.k}
% kernelParams{model.kern.nParams}
% beta{prod(size(model.beta))}


if nargout > 1
  returnNames = true;
else
  returnNames = false;
end 

if isfield(model, 'dynamics') & ~isempty(model.dynamics)
    % [VariationalParameters(reparam)   dynKernelParameters]
    if returnNames
        [dynParams, dynParamNames] = modelExtractParam(model.dynamics);%vardistExtractParam.m
        names = dynParamNames;
    else
        %%%
        dynParams = modelExtractParam(model.dynamics);%vardistExtractParam.m
    end
    params = dynParams;% add variational parameters :mu and lambda
else
    % Variational parameters 
    if returnNames
        %[varParams, varNames] = vardistExtractParam(model.vardist);
        [varParams, varNames] = modelExtractParam(model.vardist);
        %names = varNames{:}; %%% ORIGINAL
        names = varNames; %%% NEW
    else
        %varParams = vardistExtractParam(model.vardist);
        varParams = modelExtractParam(model.vardist);%% varParams contain vardist's mean and covars(lambda = exp(alpha)), here covars are alpha. 
    end
    params = varParams;
end


%if does not fix Inducing inputs 
if ~model.fixInducing 
    params =  [params model.X_v(:)'];
    params =  [params model.X_u(:)'];
    
%     for i = 1:size(model.X_v, 1)
%           for j = 1:size(model.X_v, 2)
%               X_vNames{i, j} = ['X_v(' num2str(i) ', ' num2str(j) ')'];
%           end
%     end
%       
%     if returnNames 
%       for i = 1:size(model.X_u, 1)
%           for j = 1:size(model.X_u, 2)
%               X_uNames{i, j} = ['X_u(' num2str(i) ', ' num2str(j) ')'];
%           end
%       end
%       
%       names = {names{:}, X_uNames{:},X_vNames{i, j}};
%     end
end





% beta in the likelihood 
if model.optimiseBeta
   if ~isstruct(model.betaTransform)
       fhandle = str2func([model.betaTransform 'Transform']);
       betaParam = fhandle(model.beta, 'xtoa');%% model.beta = exp(betaParam), betaParam = log(model.beta)
   else
      if isfield(model.betaTransform,'transformsettings') && ~isempty(model.betaTransform.transformsettings)
          fhandle = str2func([model.betaTransform.type 'Transform']);
          betaParam = fhandle(model.beta, 'xtoa', model.betaTransform.transformsettings);
      else
          error('vargplvmExtractParam: Invalid transform specified for beta.'); 
      end
   end   
   params = [params betaParam(:)'];% add beta into param
end

  params = [params reshape(model.W(samd,:),1,[])];% add beta into param
  
  kernParams_v = kernExtractParam(model.kern_v);
  kernParams_u = kernExtractParam(model.kern_u);
  
params = [params kernParams_v kernParams_u];

alphaParam = sigmoidTransform(model.alpha,'xtoa'); % alpha = 1/(1+exp(-alphaParam)); alphaParam = -log(1/alpha-1)
params = [params alphaParam(:)'];

%use reparameter to ensure the positiveness of epsilon
epsilonParam = expTransform(model.epsilon, 'xtoa'); % epsilon = exp(epsilonParam), epsilonParam = log(epsilon)
params = [params epsilonParam(:)'];% add beta into param


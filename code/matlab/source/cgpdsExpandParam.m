function model = cgpdsExpandParam(model, params, samd)

% cgpdsExpandParam Expand a parameter vector into a CGPDS model.
%
%	Description:
%
%	cgpdsExpandParam(MODEL, PARAMS) takes an VARGPLVM structure and a
%	vector of parameters, and fills the structure with the given
%	parameters. Also performs any necessary precomputation for
%	likelihood and gradient computations, so can be computationally
%	intensive to call.
%	 Arguments:
%	  MODEL - the CGPDS structure to put the parameters in.
%	  PARAMS - parameter vector containing the parameters to put in the
%	   VARGPLVM structure.
%	
%	

%%% Parameters must be passed as a vector in the following order (left to right) 
% - parameter{size} -
% vardistParams{model.vardist.nParams} % mu, S
%       OR
% [dynamicsVardistParams{dynamics.vardist.nParams} dynamics.kernParams{dynamics.kern.nParams}] % mu_bar, lambda
% inducingInputs{model.q*model.k}
% kernelParams{model.kern.nParams}
% beta{prod(size(model.beta))}


startVal = 1;
if isfield(model, 'dynamics') & ~isempty(model.dynamics)
    % variational parameters (reparametrized) AND dyn.kernel's parameters
    endVal = model.dynamics.nParams;
    if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
        model.dynamics = modelExpandParam(model.dynamics, params(startVal:endVal),samd);
    end
else
    % variational parameters (means and covariances), original ones
    endVal = model.vardist.nParams;
    model.vardist = modelExpandParam(model.vardist, params(startVal:endVal)); 
end



% inducing inputs 
startVal = endVal+1;
if model.fixInducing  
    if isfield(model, 'dynamics') & ~isempty(model.dynamics)
        Kt = kernCompute(model.dynamics.kern, model.dynamics.t);%%%%%%%%%%%%5
        model.X_u = Kt*model.dynamics.vardist.means; %dynamics
        model.X_u = model.X_u(model.inducingIndices,:);
        model.X_v = model.X_u;
    else
         model.X_u = model.vardist.means(model.inducingIndices, :); % static
         model.X_v = model.X_u;
    end
else
    % Parameters include inducing variables.
    endVal = endVal + model.q*model.k;
    if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
        model.X_v = reshape(params(startVal:endVal),model.k,model.q);
    end
%     model.X_v = reshape(params(startVal:endVal),model.k,model.q);
    
    startVal = endVal+1;
    endVal = endVal + model.q*model.k;
    
    if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
        model.X_u = reshape(params(startVal:endVal),model.k,model.q);
    end
%     model.X_u = reshape(params(startVal:endVal),model.k,model.q);
end



% likelihood beta parameters
if model.optimiseBeta
  startVal = endVal + 1;
  endVal = endVal + prod(size(model.beta));
  if ~isstruct(model.betaTransform)
    fhandle = str2func([model.betaTransform 'Transform']);
    
    if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
        model.beta = fhandle(params(startVal:endVal), 'atox');
    end
%     model.beta = fhandle(params(startVal:endVal), 'atox');
  else
      if isfield(model.betaTransform,'transformsettings') && ~isempty(model.betaTransform.transformsettings)
          fhandle = str2func([model.betaTransform.type 'Transform']);
          model.beta = fhandle(params(startVal:endVal), 'atox', model.betaTransform.transformsettings);
      else
          error('vargplvmExtractParam: Invalid transform specified for beta.'); 
      end
  end
end

startVal = endVal + 1;
endVal = endVal + prod(size(model.W(samd,:)));
lend = length(samd);
if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
        model.W(samd,:) = reshape(params(startVal:endVal),lend,model.J);
end
% model.W(samd,:) = reshape(params(startVal:endVal),lend,model.J);
  
% kernel hyperparameters 
startVal = endVal+1; 
endVal = endVal + model.kern_v.nParams;
if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
    model.kern_v = kernExpandParam(model.kern_v, params(startVal:endVal));
end
% model.kern_v = kernExpandParam(model.kern_v, params(startVal:endVal));

startVal = endVal+1; 
endVal = endVal + model.kern_u.nParams;

if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
    model.kern_u = kernExpandParam(model.kern_u, params(startVal:endVal));
end
% model.kern_u = kernExpandParam(model.kern_u, params(startVal:endVal));

startVal = endVal+1;
endVal = endVal + 1;
if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
    model.alpha = sigmoidTransform(params(startVal:endVal),'atox');
end

startVal = endVal+1;
endVal = endVal + 1 ;
if isempty(find(isnan(params(startVal:endVal))))&&isempty(find(isinf(params(startVal:endVal))))
    model.epsilon = expTransform(params(startVal:endVal), 'atox'); % epsilon = exp(epsilonParam), params(startVal:endVal) are epsilonParam
end

model.nParams = endVal;




% Update statistics
model = cgpdsUpdateStats(model, model.X_v ,model.X_u, samd);

% %%%TEMP: This is not needed, probably. If yes, it should be merged with
% %%%the above code for fixInducing.
% if model.fixInducing
%     model.X_u=model.X(model.inducingIndices, :);
% end
% %%%


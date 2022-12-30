function g = mvcgpdsLogLikeGradients(model, samd)

% mvcgpdsLogLikeGradients Compute the gradients for the mvcgpds.
% FORMAT
% DESC returns the gradients of the log likelihood with respect to the
% parameters of the mvcgpds model and with respect to the latent
% positions of the mvcgpds model.
% ARG model : the MVCGPDS structure containing the parameters and
% the latent positions.
% RETURN g : the gradients of the latent positions (or the back
% constraint's parameters) and the parameters of the mvcgpds model.
%

try
    p = gcp('nocreate');
    pool_open = p.NumWorkers;
    %     pool_open = matlabpool('size')>0;
catch e
    pool_open = 0;
end

if pool_open && (isfield(model,'parallel') && model.parallel)
    g=gPar(model);
else
    % Functions g1 and g2 should be equivalent for the static case, but
    % g1 might (?) be faster.
%     if ~isfield(model, 'dynamics') || isempty(model.dynamics)
%         g=g1(model);
%     else
        if isfield(model.dynamics, 'seq') & ~isempty(model.dynamics.seq)
            g = gDyn(model ,samd); % Slower but memory efficient.
        else
            g = gDynFast(model, samd); % Faster, but memory consuming
        end
%     end
end

end

%%%%%!!!!!! NOTE:
% g1 and g2 should be equivalent for the non-dynamics case. gPar is the
% equivalent of g1 for parallel computations. gDyn and gDynFast are
% suitable when there are dynamics, with the first focusing on memory
% efficiency and the second on speed.



function g = gDynFast(modelAll, samd)

gPrivAll = [];
gSharedCoeff = 0;
dynModel = modelAll.dynamics;
gSharedVar = zeros(1,dynModel.vardist.nParams);
gShareVarmeans = zeros(modelAll.N,modelAll.q);
gShareVarcovars = cell(modelAll.q,1);

% for q = 1:modelAll.q
%     gShareVarmeans(:,q) = (1-modelAll.comp{1}.alpha)^2*inv(modelAll.comp{1}.dynamics.At)*dynModel.vardist.means(:,q)...
%         + (1-modelAll.comp{2}.alpha)^2*inv(modelAll.comp{2}.dynamics.At)*dynModel.vardist.means(:,q) ...
%         + inv(dynModel.Kt)*dynModel.vardist.means(:,q)...
%         - (1-modelAll.comp{1}.alpha)*inv(modelAll.comp{1}.dynamics.At)*modelAll.comp{1}.dynamics.vardist.means(:,q)...
%         - (1-modelAll.comp{2}.alpha)*inv(modelAll.comp{2}.dynamics.At)*modelAll.comp{2}.dynamics.vardist.means(:,q);
%     
%     
%     gShareVarcovars{q} = -0.5*inv(dynModel.vardist.Sq{q}) ...
%         + 0.5*((1-modelAll.comp{1}.alpha)^2*inv(modelAll.comp{1}.dynamics.At) ...
%         + (1-modelAll.comp{2}.alpha)^2*inv(modelAll.comp{2}.dynamics.At) + inv(dynModel.Kt));
% end


for m=1:modelAll.numModels
    % This is similar to vargplvmLogLikeGradients but for every model
    
    model = modelAll.comp{m}; % current model
    samd = [1:model.d];
    
    % Likelihood terms (coefficients)
    [gK_vv , gK_uu, gPsi0, gPsi1, gPsi2,gPsi3, gPsi4, gPsi5, g_Lambda, gBeta, gW , gepsilon,  tmpV_v ,tmpV_u] = vargpCovGrads(model ,dynModel, modelAll, samd);
    
    if isfield(model, 'learnInducing')
        learnInducing = model.learnInducing;
    else
        learnInducing = true;
    end
    
    % Get (in three steps because the formula has three terms) the gradients of
    % the likelihood part w.r.t the data kernel parameters, variational means
    % and covariances (original ones). From the field model.vardist, only
    % vardist.means and vardist.covars and vardist.lantentDimension are used.
    [gKern0, gVarmeans0, gVarcovs0, gInd0] = kernVardistPsi1Gradient(model.kern_v, model.vardist, model.X_v, gPsi0');
    [gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model.kern_u, model.vardist, model.X_u, gPsi1');
   
    [gKern2, gVarmeans2, gVarcovs2] = kernVardistPsi0Gradient(model.kern_v, model.vardist, gPsi2);
    [gKern3, gVarmeans3, gVarcovs3] = kernVardistPsi0Gradient(model.kern_u, model.vardist, gPsi3);
    
    [gKern4, gVarmeans4, gVarcovs4, gInd4] = kernVardistPsi2Gradient(model.kern_v, model.vardist, model.X_v, gPsi4);
    [gKern5, gVarmeans5, gVarcovs5, gInd5] = kernVardistPsi2Gradient(model.kern_u, model.vardist, model.X_u, gPsi5);
 
    gKern6 = kernGradient(model.kern_v, model.X_v, gK_vv);
    gKern7 = kernGradient(model.kern_u, model.X_u, gK_uu);
    
    % At this point, gKern gVarmeansLik and gVarcovsLik have the derivatives for the
    % likelihood part. Sum all of them to obtain the final result.
    gKern_v = gKern0 + gKern2 + gKern4 + gKern6;
    gKern_u = gKern1 + gKern3 + gKern5 + gKern7;
    gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2 + gVarmeans3 + gVarmeans4 + gVarmeans5;
    
    % if strcmp(model.kern.type, 'rbfardjit')
    %     % different derivatives for the variance, which is super-numerically stable for
    %     % this particular kernel
    %     gKern(1) = 0.5*model.d*( - model.k+ sum(sum(model.invLat.*model.invLat))/model.beta - model.beta*(model.Psi0-model.TrC)  )...
    %                     + 0.5*tmpV;
    % end
    
    % if strcmp(model.kern.type, 'rbfardjit')
    %     % different derivatives for the variance, which is super-numerically stable for
    %     % this particular kernel
    %     if model.learnSigmaf == 1
    %         gKern(1) = 0.5*model.d*( - model.k+ sum(sum(model.invLat.*model.invLat))/model.beta - model.beta*(model.Psi0-model.TrC)  )...
    %             + 0.5*tmpV;
    %
    %         if ~isstruct(model.kern.transforms(1))
    %             fhandle = str2func([model.kern.transform(1) 'Transform']);
    %             gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
    %         else
    %             fhandle = str2func([model.kern.transforms(1).type 'Transform']);
    %             if ~isfield(model.kern.transforms(1), 'transformsettings')
    %                 gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
    %             else
    %                 gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact', model.kern.transforms(1).transformsettings);
    %             end
    %         end
    %     else
    %         gKern(1) = 0;
    %     end
    % end
    
    
    %%% Compute Kvv and Kuu Gradients with respect to X_v and X_u %%%
    gKX_v = kernGradX(model.kern_v, model.X_v, model.X_v);
    gKX_u = kernGradX(model.kern_u, model.X_u, model.X_u);
    
    % The 2 accounts for the fact that covGrad is symmetric
    gKX_v = gKX_v*2;
    dgKX_v = kernDiagGradX(model.kern_v, model.X_v);
    for i = 1:model.k
        gKX_v(i, :, i) = dgKX_v(i, :);
    end
    
    gKX_u = gKX_u*2;
    dgKX_u = kernDiagGradX(model.kern_u, model.X_u);
    for i = 1:model.k
        gKX_u(i, :, i) = dgKX_u(i, :);
    end
    
    
    % Allocate space for gX_u
    gX_u = zeros(model.k, model.q);
    % Compute portion associated with gK_u
    for i = 1:model.k
        for j = 1:model.q
            gX_u(i, j) = gKX_u(:, j, i)'*gK_uu(:, i);%gKuu Q*M
        end
    end
    
    % Allocate space for gX_u
    gX_v = zeros(model.k, model.q);
    % Compute portion associated with gK_u
    for i = 1:model.k
        for j = 1:model.q
            gX_v(i, j) = gKX_v(:, j, i)'*gK_vv(:, i);
        end
    end
    
    gKern_u= [];
    
    % This should work much faster
    %gX_u2 = kernKuuXuGradient(model.kern, model.X_u, gK_uu);
    
    %sum(abs(gX_u2(:)-gX_u(:)))
    %pause
    
    %the gradients of the inducing points
    gInd_v = gInd0 + gInd4 + gX_v(:)';
    gInd_u = gInd1 + gInd5 + gX_u(:)';
    

    
    
    % If we only want to exclude the derivatives for the variational
    % distribution, the following big block will be skipped.
    if ~(isfield(model, 'onlyKernel') && model.onlyKernel)
%         if isfield(model, 'dynamics') && ~isempty(model.dynamics)
            % Calculate the derivatives for the reparametrized variational and Kt parameters.
            % The formulae for these include in a mixed way the derivatives of the KL
            % term w.r.t these., so gVarmeansKL and gVarcovsKL are not needed now. Also
            % the derivatives w.r.t kernel parameters also require the derivatives of
            % the likelihood term w.r.t the var. parameters, so this call must be put
            % in this part.
            
            % For the dynamical model further the original covs. must be fed,
            % before amending with the partial derivative due to exponing to enforce
            % positiveness.
            gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
            
            % gVarmeansLik and gVarcovsLik are serialized into 1x(NxQ) vectors.
            % Convert them to NxQ matrices, i.e. fold them column-wise.
            gVarmeansLik = reshape(gVarmeansLik,model.N,model.q);
            gVarmeans = gVarmeansLik - inv(model.dynamics.At)*(model.dynamics.vardist.means - (1-model.alpha)*dynModel.vardist.means);
            
            gVarcovsLik = reshape(gVarcovsLik,model.N,model.q);
            
            
            gVarcovs = zeros(model.N, model.q); % memory preallocation
            sumTrGradKL = 0;
            trGradKL = 0;
            
            for q=1:model.q
                LambdaH_q = model.dynamics.vardist.covars(:,q).^0.5;
                Bt_q = eye(model.N) + LambdaH_q*LambdaH_q'.*model.dynamics.At;
                
                % Invert Bt_q
                Lbt_q = jitChol(Bt_q)';
                G1 = Lbt_q \ diag(LambdaH_q);
                G = G1*model.dynamics.At;
                
                % Find Sq
                Sq = model.dynamics.At - G'*G;
                
                % If onlyLikelihood is set to 1, then the KL part will be multiplied
                % with zero, thus being ignored.
                gVarcovs(:,q) = - (Sq .* Sq) * (gVarcovsLik(:,q) + ...
                    0.5*model.dynamics.vardist.covars(:,q));
                
                % Find the coefficient for the grad. wrt theta_t (params of Kt)
                G1T=G1';
                Bhat=G1T*G1;
                BhatAt=G1T*G;
                
                % If we only need the derivatives from the likelihood part of the
                % bound, set the corrsponding KL part to zero.
                
%                 trGradKL = -0.5*(BhatAt*Bhat - inv(model.dynamics.At)*((1-model.alpha)*dynModel.vardist.means(:,q)...
%                     -model.dynamics.vardist.means(:,q))*((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'...
%                     *inv(model.dynamics.At));

                IBA = eye(model.N) - BhatAt;
                diagVarcovs = repmat(gVarcovsLik(:,q)', model.N,1);
                trGradKL = IBA .*  diagVarcovs * IBA';
                diagLambda = repmat(model.dynamics.vardist.covars(:,q)', model.N, 1);
                invAt = inv(model.dynamics.At);
                trGradKL = trGradKL -0.5*(invAt - invAt*((1-model.alpha)*dynModel.vardist.means(:,q)...
                     -model.dynamics.vardist.means(:,q))*((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'*invAt...
                     - invAt*dynModel.vardist.Sq{q}*invAt*(1-model.alpha)^2 - invAt*model.dynamics.vardist.Sq{q}*invAt...
                     - IBA .* diagLambda * IBA');
                 
                trGradKL = trGradKL*(model.alpha)^2;
                % In case gInd is empty then the inducing points are not reparametrized
                % (are not fixed to the variational means) we need not amend further the
                % derivative w.r.t theta_t, otherwise we have to do that.
%                 if ~isempty(gInd_v)
%                     trGradKL = trGradKL + dynModel.vardist.means(:,q) * gInd_v(:,q)';
%                 end
%                 if ~isempty(gInd_u)
%                     trGradKL = trGradKL + dynModel.vardist.means(:,q) * gInd_u(:,q)';
%                 end
                
                sumTrGradKL = sumTrGradKL + trGradKL;
            end
            gDynKern = kernGradient(model.dynamics.kern, model.dynamics.t, sumTrGradKL);
            
            % Serialize (unfold column-wise) gVarmeans and gVarcovs from NxQ matrices
            % to 1x(NxQ) vectors
            gVarcovs = gVarcovs(:)';
            gVarmeans = gVarmeans(:)';
            %%%[gVarmeans gVarcovs gDynKern] = modelPriorReparamGrads(model.dynamics, gVarmeansLik, gVarcovsLik, gIndRep_v , gIndRep_u);
            % Variational variances are positive: Now that the final covariances
            % are obtained we amend with the partial derivatives due to the
            % exponential transformation to ensure positiveness.
            if ~isfield(model, 'notransform') || (isfield(model,'notransform') && model.notransform == false)
                gVarcovs = (gVarcovs(:).*model.dynamics.vardist.covars(:))';
            end
            
            galpha = 0;
            gLikalpha = 0;
            gKLalpha = 0;
            if ~(isfield(model,'fixAlpha') && model.fixAlpha == 1)
                for q=1:model.q
                    gLikalpha = gLikalpha + trace(gVarcovsLik(:,q)'*diag(model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(2*model.alpha*model.dynamics.Kt...
                        -2*(1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}));
                          
                    gKLalpha = gKLalpha + trace(inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N))) ...
                        - trace(inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q})...
                        - trace(inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt -(1-model.alpha)*model.epsilon*eye(model.N)))...
                        + trace(inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt -(1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q})...
                        - trace((1-model.alpha)*inv(model.dynamics.At)*dynModel.vardist.Sq{q})...
                        - (1-model.alpha)^2*trace(inv(model.dynamics.At)*dynModel.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N)))...
                        - dynModel.vardist.means(:,q)'*inv(model.dynamics.At)*((1-model.alpha)*dynModel.vardist.means(:,q) - model.dynamics.vardist.means(:,q))...
                        - trace(inv(model.dynamics.At)*((1-model.alpha)*dynModel.vardist.means(:,q) - model.dynamics.vardist.means(:,q))*((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N)));
                      
                end
                galpha = gLikalpha - gKLalpha;
            end
            galphaParam = galpha*model.alpha*(1-model.alpha); 
%         else
%             % For the non-dynamical GPLVM these cov. derivatives are the final, so
%             % it is time to amend with the partial derivative due to exponing them
%             % to force posigiveness.
%             gVarcovs0 = (gVarcovs0(:).*model.vardist.covars(:))';
%             gVarcovs1 = (gVarcovs1(:).*model.vardist.covars(:))';
%             gVarcovs2 = (gVarcovs2(:).*model.vardist.covars(:))';
%             gVarcovs3 = (gVarcovs3(:).*model.vardist.covars(:))';
%             gVarcovs4 = (gVarcovs4(:).*model.vardist.covars(:))';
%             gVarcovs5 = (gVarcovs5(:).*model.vardist.covars(:))';
%             
%             gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
%             gVarmeans = gVarmeansLik + gVarmeansKL;
%             %gVarcovsLik = (gVarcovsLik(:).*model.vardist.covars(:))';
%             gVarcovs = gVarcovsLik + gVarcovsKL;
%         end
    else
        gVarmeans = [];
        gVarcovs = [];
        gDynKern = [];
    end
    
    gInd_u = [];
%     if isfield(model, 'fixInducing') & model.fixInducing
%         % If there are dynamics the derivative must further be amended with the
%         % partial deriv. due to the mean reparametrization.
%         if isfield(model, 'dynamics') && ~isempty(model.dynamics)
%             gInd_v = reshape(gInd_v,model.k,model.q);
%             %gInd = gInd' * model.dynamics.Kt;
%             gInd_v =  model.dynamics.Kt * gInd_v;
%             gInd_v = gInd_v(:)';
%             
%             gInd_u = reshape(gInd_u,model.k,model.q);
%             %gInd = gInd' * model.dynamics.Kt;
%             gInd_u =  model.dynamics.Kt * gInd_u;
%             gInd_u = gInd_u(:)';
%         end
%         %gVarmeans(model.inducingIndices, :) = gVarmeans(model.inducingIndices,
%         %:) + gInd; % This should work AFTER reshaping the matrices...but here
%         %we use all the indices anyway.
%         gVarmeans = gVarmeans + gInd_v + gInd_u;
%         gInd_v = []; % Inducing points are not free variables anymore, they dont have derivatives on their own.
%         gInd_u = [];
%     end
    
    gVar = [gVarmeans gVarcovs];
    
    if isfield(model.vardist,'paramGroups')
        gVar = gVar*model.vardist.paramGroups;
    end
    
    
    % If we only want to exclude the derivatives for the variational
    % distribution, the following big block will be skipped.
    if ~(isfield(model, 'onlyKernel') && model.onlyKernel)
        % It may better to start optimize beta a bit later so that
        % so that the rest parameters can be initialized
        % (this could help to escape from the trivial local
        % minima where the noise beta explains all the data)
        % The learnBeta option deals with the above.
        
        % This constrains the variance of the dynamics kernel to one
        % (This piece of code needs to be done in better way with unit variance dynamic
        %  kernels. The code below also will only work for rbf dynamic kernel)
        % Assume that the base rbf/matern/etc kernel is first in the compound
        % structure
        if isfield(model, 'dynamics') && ~isempty(model.dynamics)
            if strcmp(model.dynamics.kern.comp{1}.type,'rbf') || strcmp(model.dynamics.kern.comp{1}.type,'matern32') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic2')
                if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
                    gDynKern(2) = 0;
                end
            end
            
            %___NEW: assume that the second rbf/matern etc kernel is last in the
            %compound kernel
            %if numel(model.dynamics.kern.comp) > 3
            if isfield(model.dynamics, 'learnSecondVariance') && ~model.dynamics.learnSecondVariance   %%%%% NEW
                gDynKern(end) = 0;
            end
            %end
            %___
        end
    end
    
    if isempty(gVar)
        gVar = zeros(1,model.q*model.N*2);
    end
    if isempty(gInd_v)
        gInd_v = 0*model.X_v(:)';
    end
    if isempty(gInd_u)
        gInd_u = 0*model.X_u(:)';
    end
    if isempty(gKern_u)
        gKern_u = zeros(1,model.kern_u.nParams);
    end
    if isempty(gKern_v)
        gKern_v = zeros(1,model.kern_v.nParams);
    end
    if isempty(gDynKern)
        gDynKern= zeros(1,model.dynamics.kern.nParams);
    end
    %%
    % In case we are in the phase where the vardistr. is initialised (see above
    % for the variance of the kernel), beta is kept fixed. For backwards
    % compatibility this can be controlled either with the learnBeta field or
    % with the initVardist field. The later overrides the first.
    
     if isfield(model, 'learnBeta') && model.learnBeta
        gBetaFinal = gBeta;
    else
        gBetaFinal = 0*gBeta;
     end
    
    if isfield(modelAll, 'onlyOptiVardist') && modelAll.onlyOptiVardist == 1 
        gDynKern = 0*gDynKern;
        gBetaFinal = 0*gBetaFinal;
        gW = 0*gW;
        gKern_v = 0*gKern_v;
        gKern_u = 0*gKern_u;
        galpha = 0*galpha;
        gepsilon = 0*gepsilon;
    end
    if isfield(modelAll, 'onlyOptiModel') && modelAll.onlyOptiModel == 1 
        gVar = 0*gVar;
        gInd_v = 0*gInd_v;
        gInd_u = 0*gInd_u;
    end
    
    if isfield(modelAll, 'fixW') && modelAll.fixW == 1 
        gW = 0*gW;
    end
    if isfield(modelAll, 'fixEpsilon') && modelAll.fixEpsilon == 1 
        gepsilon = 0*gepsilon;
    end
    if isfield(modelAll, 'fixDynKern') && modelAll.fixDynKern == 1 
        gDynKern = 0*gDynKern;
    end
    if isfield(modelAll, 'fixBeta') && modelAll.fixBeta == 1 
        gBetaFinal = 0*gBetaFinal;
    end
    if isfield(modelAll, 'fixAlpha') && modelAll.fixAlpha == 1 
        galphaParam = 0*galphaParam;
    end
    
    
    g = [gVar gDynKern gInd_v gInd_u gBetaFinal gW(:)' gKern_v gKern_u galphaParam gepsilon];
    
    % At this point, gDynKern will be [] if there are no dynamics]
    gPrivAll = [gPrivAll g];
    
    
%     if strcmp(model.kern.type, 'rbfardjit')
%         % different derivatives for the variance, which is super-numerically stable for
%         % this particular kernel
%         if model.learnSigmaf == 1
%             gKern(1) = 0.5*model.d*( - model.k+ sum(sum(model.invLat.*model.invLat))/model.beta - model.beta*(model.Psi0-model.TrC)  )...
%                 + 0.5*tmpV;
%             
%             if ~isstruct(model.kern.transforms(1))
%                 fhandle = str2func([model.kern.transform(1) 'Transform']);
%                 gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%             else
%                 fhandle = str2func([model.kern.transforms(1).type 'Transform']);
%                 if ~isfield(model.kern.transforms(1), 'transformsettings')
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%                 else
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact', model.kern.transforms(1).transformsettings);
%                 end
%             end
%         else
%             gKern(1) = 0;
%         end
%     end
    
    
end

gShareDynKernCoeff = 0;
for q=1:model.q
    gShareDynKernCoeff = gShareDynKernCoeff + 0.5*(inv(dynModel.Kt) - inv(dynModel.Kt)*(dynModel.vardist.means(:,q)...
        *dynModel.vardist.means(:,q)' + dynModel.vardist.Sq{q})*inv(dynModel.Kt));
end
gShareDynKern = kernGradient(dynModel.kern, dynModel.t, gShareDynKernCoeff);


%%%% ...... was commented...
if isfield(model, 'dynamics') && ~isempty(model.dynamics)
    if strcmp(model.dynamics.kern.comp{1}.type,'rbf') || strcmp(model.dynamics.kern.comp{1}.type,'matern32') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic2')
        if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
            gShareDynKern(2) = 0;
        end
    end
    
    %___NEW: assume that the second rbf/matern etc kernel is last in the
    %compound kernel
    %if numel(model.dynamics.kern.comp) > 3
    if isfield(model.dynamics, 'learnSecondVariance') && ~model.dynamics.learnSecondVariance   %%%%% NEW
        gShareDynKern(end) = 0;
    end
    %end
    %___
end


%---- NEW 2012: This is to fix selectively some of the kernel's parameters
if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
    % The field model.dynamics.fixedVariance must have the
    % indexes of the gradient vector that correspond to
    % variances of kernels of the matern class and we wish to
    % not learn them.
    if isfield(model.dynamics, 'fixedKernVariance') && ~isempty(model.dynamics.fixedKernVariance)
        gShareDynKern(model.dynamics.fixedKernVariance) = 0;
    end
end
%----

if isfield(modelAll, 'fixShareDynKern') && modelAll.fixShareDynKern == 1 
        gShareDynKern = 0*gShareDynKern;
end
    
g = [gSharedVar gShareDynKern gPrivAll];

end

function g = gDyn(modelAll, samd)

gPrivAll = [];
gSharedCoeff = 0;
dynModel = modelAll.dynamics;
gSharedVar = zeros(1,dynModel.vardist.nParams);
% gShareVarmeans = zeros(modelAll.N,modelAll.q);
% gShareVarcovars = cell(modelAll.q,1);
% for q = 1:modelAll.q
%     gShareVarmeans(:,q) = (1-modelAll.comp{1}.alpha)^2*inv(modelAll.comp{1}.dynamics.At)*dynModel.vardist.means(:,q)...
%         + (1-modelAll.comp{2}.alpha)^2*inv(modelAll.comp{2}.dynamics.At)*dynModel.vardist.means(:,q) ...
%         + inv(dynModel.Kt)*dynModel.vardist.means(:,q)...
%         - (1-modelAll.comp{1}.alpha)*inv(modelAll.comp{1}.dynamics.At)*modelAll.comp{1}.dynamics.vardist.means(:,q)...
%         - (1-modelAll.comp{2}.alpha)*inv(modelAll.comp{2}.dynamics.At)*modelAll.comp{2}.dynamics.vardist.means(:,q);
%     
%     
%     gShareVarcovars{q} = -0.5*inv(dynModel.vardist.Sq{q}) ...
%         + 0.5*((1-modelAll.comp{1}.alpha)^2*inv(modelAll.comp{1}.dynamics.At) ...
%         + (1-modelAll.comp{2}.alpha)^2*inv(modelAll.comp{2}.dynamics.At) + inv(dynModel.Kt));
% end

if isfield(dynModel, 'seq') & ~isempty(dynModel.seq)
    seqStart=1;
    seq = dynModel.seq;
    for i=1:length(dynModel.seq)
        seqEnd = seq(i);
        sumTrGradKL{i} = zeros(seqEnd-seqStart+1, seqEnd-seqStart+1);
        seqStart = seqEnd+1;
    end
end


for m=1:modelAll.numModels
    % This is similar to vargplvmLogLikeGradients but for every model
    
    model = modelAll.comp{m}; % current model
    samd = [1:model.d];
    
    % Likelihood terms (coefficients)
    [gK_vv , gK_uu, gPsi0, gPsi1, gPsi2,gPsi3, gPsi4, gPsi5, g_Lambda, gBeta, gW , gepsilon, tmpV_v ,tmpV_u] = vargpCovGrads(model ,dynModel, modelAll, samd);
    
    if isfield(model, 'learnInducing')
        learnInducing = model.learnInducing;
    else
        learnInducing = true;
    end
    
    % Get (in three steps because the formula has three terms) the gradients of
    % the likelihood part w.r.t the data kernel parameters, variational means
    % and covariances (original ones). From the field model.vardist, only
    % vardist.means and vardist.covars and vardist.lantentDimension are used.
    [gKern0, gVarmeans0, gVarcovs0, gInd0] = kernVardistPsi1Gradient(model.kern_v, model.vardist, model.X_v, gPsi0');
    [gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model.kern_u, model.vardist, model.X_u, gPsi1');
   
    [gKern2, gVarmeans2, gVarcovs2] = kernVardistPsi0Gradient(model.kern_v, model.vardist, gPsi2);
    [gKern3, gVarmeans3, gVarcovs3] = kernVardistPsi0Gradient(model.kern_u, model.vardist, gPsi3);
    
    [gKern4, gVarmeans4, gVarcovs4, gInd4] = kernVardistPsi2Gradient(model.kern_v, model.vardist, model.X_v, gPsi4);
    [gKern5, gVarmeans5, gVarcovs5, gInd5] = kernVardistPsi2Gradient(model.kern_u, model.vardist, model.X_u, gPsi5);
 
    gKern6 = kernGradient(model.kern_v, model.X_v, gK_vv);
    gKern7 = kernGradient(model.kern_u, model.X_u, gK_uu);
    
    % At this point, gKern gVarmeansLik and gVarcovsLik have the derivatives for the
    % likelihood part. Sum all of them to obtain the final result.
    gKern_v = gKern0 + gKern2 + gKern4 + gKern6;
    gKern_u = gKern1 + gKern3 + gKern5 + gKern7;
    gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2 + gVarmeans3 + gVarmeans4 + gVarmeans5;
    
%     if strcmp(model.kern_v.type, 'rbfardjit')
%         % different derivatives for the variance, which is super-numerically stable for
%         % this particular kernel
%         if model.learnSigmaf == 1
%             gKern(1) = 0.5*model.d*( - model.k+ sum(sum(model.invLat.*model.invLat))/model.beta - model.beta*(model.Psi0-model.TrC)  )...
%                 + 0.5*tmpV;
%     
%             if ~isstruct(model.kern.transforms(1))
%                 fhandle = str2func([model.kern.transform(1) 'Transform']);
%                 gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%             else
%                 fhandle = str2func([model.kern.transforms(1).type 'Transform']);
%                 if ~isfield(model.kern.transforms(1), 'transformsettings')
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%                 else
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact', model.kern.transforms(1).transformsettings);
%                 end
%             end
%         else
%             gKern(1) = 0;
%         end
%     end
    
    %%% Compute Kvv and Kuu Gradients with respect to X_v and X_u %%%
    gKX_v = kernGradX(model.kern_v, model.X_v, model.X_v);
    gKX_u = kernGradX(model.kern_u, model.X_u, model.X_u);
    
    % The 2 accounts for the fact that covGrad is symmetric
    gKX_v = gKX_v*2;
    dgKX_v = kernDiagGradX(model.kern_v, model.X_v);
    for i = 1:model.k
        gKX_v(i, :, i) = dgKX_v(i, :);
    end
    
    gKX_u = gKX_u*2;
    dgKX_u = kernDiagGradX(model.kern_u, model.X_u);
    for i = 1:model.k
        gKX_u(i, :, i) = dgKX_u(i, :);
    end
    
    
    % Allocate space for gX_u
    gX_u = zeros(model.k, model.q);
    % Compute portion associated with gK_u
    for i = 1:model.k
        for j = 1:model.q
            gX_u(i, j) = gKX_u(:, j, i)'*gK_uu(:, i);%gKuu Q*M
        end
    end
    
    % Allocate space for gX_u
    gX_v = zeros(model.k, model.q);
    % Compute portion associated with gK_u
    for i = 1:model.k
        for j = 1:model.q
            gX_v(i, j) = gKX_v(:, j, i)'*gK_vv(:, i);
        end
    end
   
    gKern_u = [];
    
    %the gradients of the inducing points
    gInd_v = gInd0 + gInd4 + gX_v(:)';
    gInd_u = gInd1 + gInd5 + gX_u(:)';
    
%     if isfield(model, 'fixInducing') & model.fixInducing
%         gIndRep_v = gInd_v;
%         gIndRep_u = gInd_u;
%     else
%         gIndRep_v=[];
%         gIndRep_u=[];
%     end
    
    % If we only want to exclude the derivatives for the variational
    % distribution, the following big block will be skipped.
    if ~(isfield(model, 'onlyKernel') && model.onlyKernel)
%         if isfield(model, 'dynamics') && ~isempty(model.dynamics)
            % Calculate the derivatives for the reparametrized variational and Kt parameters.
            % The formulae for these include in a mixed way the derivatives of the KL
            % term w.r.t these., so gVarmeansKL and gVarcovsKL are not needed now. Also
            % the derivatives w.r.t kernel parameters also require the derivatives of
            % the likelihood term w.r.t the var. parameters, so this call must be put
            % in this part.
            
            % For the dynamical GPLVM further the original covs. must be fed,
            % before amending with the partial derivative due to exponing to enforce
            % positiveness.
            gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
            
            % gVarmeansLik and gVarcovsLik are serialized into 1x(NxQ) vectors.
            % Convert them to NxQ matrices, i.e. fold them column-wise.
            gVarmeansLik = reshape(gVarmeansLik,model.N,model.q);
            gVarmeans = gVarmeansLik - inv(model.dynamics.At)*(model.dynamics.vardist.means - (1-model.alpha)*dynModel.vardist.means);
            
            gVarcovsLik = reshape(gVarcovsLik,model.N,model.q);
            
            
            gVarcovs = zeros(model.N, model.q); % memory preallocation
            gDynKern = size(model.dynamics.kern.nParams,1);
            
            if isfield(model.dynamics, 'seq') & ~isempty(model.dynamics.seq)
                for q=1:model.q
                    LambdaH_q = model.dynamics.vardist.covars(:,q).^0.5;
                    seqStart=1;
                    for i=1:length(model.dynamics.seq)
                        seqEnd = seq(i);
                        Bt_q = eye(seqEnd-seqStart+1) + LambdaH_q(seqStart:seqEnd,1)*LambdaH_q(seqStart:seqEnd,1)'.*model.dynamics.At(seqStart:seqEnd,seqStart:seqEnd);
                        
                        % Invert Bt_q
                        Lbt_q = jitChol(Bt_q)';
                        G1 = Lbt_q \ diag(LambdaH_q(seqStart:seqEnd,1));
                        G = G1*model.dynamics.At(seqStart:seqEnd, seqStart:seqEnd);
                        
                        % Find Sq
                        Sq = model.dynamics.At(seqStart:seqEnd, seqStart:seqEnd) - G'*G;
                        
                        gVarcovs(seqStart:seqEnd,q) = - (Sq .* Sq) * (gVarcovsLik(seqStart:seqEnd,q) + ...
                            0.5*dynModel.vardist.covars(seqStart:seqEnd,q));
                        
                        % Find the coefficient for the grad. wrt theta_t (params of Kt)
                        G1T=G1';
                        Bhat=G1T*G1;
                        BhatAt=G1T*G;
                        
                        IBA = eye(seqEnd-seqStart+1) - BhatAt;
                        diagVarcovs = repmat(gVarcovsLik(seqStart:seqEnd,q)', seqEnd-seqStart+1,1);
                        trGradKL = IBA .*  diagVarcovs * IBA';
                        diagLambda = repmat(model.dynamics.vardist.covars(seqStart:seqEnd,q)', seqEnd-seqStart+1,1);
                        invAt = inv(model.dynamics.At(seqStart:seqEnd, seqStart:seqEnd));
                        trGradKL = trGradKL -0.5*(invAt - invAt*((1-model.alpha)*dynModel.vardist.means(seqStart:seqEnd,q)...
                            -model.dynamics.vardist.means(seqStart:seqEnd,q))*((1-model.alpha)*dynModel.vardist.means(seqStart:seqEnd,q)-model.dynamics.vardist.means(seqStart:seqEnd,q))'*invAt...
                            - invAt*dynModel.vardist.Sq{q}(seqStart:seqEnd, seqStart:seqEnd)*invAt*(1-model.alpha)^2 - invAt*model.dynamics.vardist.Sq{q}(seqStart:seqEnd, seqStart:seqEnd)*invAt...
                            - IBA .* diagLambda * IBA');
                        
                        trGradKL = trGradKL*(model.alpha)^2;
                        sumTrGradKL{i} = sumTrGradKL{i} + trGradKL;
                        gDynKern = gDynKern + kernGradient(model.dynamics.kern, model.dynamics.t(seqStart:seqEnd), sumTrGradKL{i});
                        seqStart = seqEnd+1;
                    end
                end
            else
                sumTrGradKL = 0;
                for q=1:model.q
                    LambdaH_q = model.dynamics.vardist.covars(:,q).^0.5;
                    Bt_q = eye(model.N) + LambdaH_q*LambdaH_q'.*model.dynamics.At;
                    
                    % Invert Bt_q
                    Lbt_q = jitChol(Bt_q)';
                    G1 = Lbt_q \ diag(LambdaH_q);
                    G = G1*model.dynamics.At;
                    
                    % Find Sq
                    Sq = model.dynamics.At - G'*G;
                    
                    % If onlyLikelihood is set to 1, then the KL part will be multiplied
                    % with zero, thus being ignored.
                    gVarcovs(:,q) = - (Sq .* Sq) * (gVarcovsLik(:,q) + ...
                        0.5*model.dynamics.vardist.covars(:,q));
                    
                    % Find the coefficient for the grad. wrt theta_t (params of Kt)
                    G1T=G1';
                    Bhat=G1T*G1;
                    BhatAt=G1T*G;
                    
                    IBA = eye(model.N) - BhatAt;
                    diagVarcovs = repmat(gVarcovsLik(:,q)', model.N,1);
                    trGradKL = IBA .*  diagVarcovs * IBA';
                    diagLambda = repmat(model.dynamics.vardist.covars(:,q)', model.N, 1);
                    invAt = inv(model.dynamics.At);
                    trGradKL = trGradKL -0.5*(invAt - invAt*((1-model.alpha)*dynModel.vardist.means(:,q)...
                        -model.dynamics.vardist.means(:,q))*((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'*invAt...
                        - invAt*dynModel.vardist.Sq{q}*invAt*(1-model.alpha)^2 - invAt*model.dynamics.vardist.Sq{q}*invAt...
                        - IBA .* diagLambda * IBA');
                    trGradKL = trGradKL*(model.alpha)^2;
                    sumTrGradKL = sumTrGradKL + trGradKL;
                end
                gDynKern = kernGradient(model.dynamics.kern, model.dynamics.t, sumTrGradKL);
            end
            
            
            % Serialize (unfold column-wise) gVarmeans and gVarcovs from NxQ matrices
            % to 1x(NxQ) vectors
            gVarcovs = gVarcovs(:)';
            gVarmeans = gVarmeans(:)';
            %%%[gVarmeans gVarcovs gDynKern] = modelPriorReparamGrads(model.dynamics, gVarmeansLik, gVarcovsLik, gIndRep_v , gIndRep_u);
            % Variational variances are positive: Now that the final covariances
            % are obtained we amend with the partial derivatives due to the
            % exponential transformation to ensure positiveness.
            if ~isfield(model, 'notransform') || (isfield(model,'notransform') && model.notransform == false)
                gVarcovs = (gVarcovs(:).*model.dynamics.vardist.covars(:))';
            end
            
            galpha = 0;
            gLikalpha = 0;
            gKLalpha = 0;
            if ~(isfield(model,'fixAlpha') && model.fixAlpha == 1)
                for q=1:model.q
                    gLikalpha = gLikalpha + trace(gVarcovsLik(:,q)'*diag(model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(2*model.alpha*model.dynamics.Kt...
                        -2*(1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}));
                          
                    gKLalpha = gKLalpha + trace(inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N))) ...
                        -trace(inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q})...
                        - trace(inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt -(1-model.alpha)*model.epsilon*eye(model.N)))...
                        + trace(inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt -(1-model.alpha)*model.epsilon*eye(model.N))*inv(model.dynamics.At)*model.dynamics.vardist.Sq{q})...
                        - trace((1-model.alpha)*inv(model.dynamics.At)*dynModel.vardist.Sq{q})...
                        - (1-model.alpha)^2*trace(inv(model.dynamics.At)*dynModel.vardist.Sq{q}*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N)))...
                        - dynModel.vardist.means(:,q)'*inv(model.dynamics.At)*((1-model.alpha)*dynModel.vardist.means(:,q) - model.dynamics.vardist.means(:,q))...
                        - trace(inv(model.dynamics.At)*((1-model.alpha)*dynModel.vardist.means(:,q) - model.dynamics.vardist.means(:,q))*((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'*inv(model.dynamics.At)*(model.alpha*model.dynamics.Kt - (1-model.alpha)*model.epsilon*eye(model.N)));
                      
                end
                galpha = gLikalpha - gKLalpha;
                galphaParam = galpha*model.alpha*(1-model.alpha);
            end
            
            
%         else
%             % For the non-dynamical these cov. derivatives are the final, so
%             % it is time to amend with the partial derivative due to exponing them
%             % to force posigiveness.
%             gVarcovs0 = (gVarcovs0(:).*model.vardist.covars(:))';
%             gVarcovs1 = (gVarcovs1(:).*model.vardist.covars(:))';
%             gVarcovs2 = (gVarcovs2(:).*model.vardist.covars(:))';
%             gVarcovs3 = (gVarcovs3(:).*model.vardist.covars(:))';
%             gVarcovs4 = (gVarcovs4(:).*model.vardist.covars(:))';
%             gVarcovs5 = (gVarcovs5(:).*model.vardist.covars(:))';
%             
%             gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2 + gVarcovs3 + gVarcovs4 + gVarcovs5;
%             gVarmeans = gVarmeansLik + gVarmeansKL;
%             %gVarcovsLik = (gVarcovsLik(:).*model.vardist.covars(:))';
%             gVarcovs = gVarcovsLik + gVarcovsKL;
%         end
    else
        gVarmeans = [];
        gVarcovs = [];
        gDynKern = [];
    end
    
   
%     if isfield(model, 'fixInducing') & model.fixInducing
%         % If there are dynamics the derivative must further be amended with the
%         % partial deriv. due to the mean reparametrization.
%         if isfield(model, 'dynamics') && ~isempty(model.dynamics)
%             gInd_v = reshape(gInd_v,model.k,model.q);
%             %gInd = gInd' * model.dynamics.Kt;
%             gInd_v =  model.dynamics.Kt * gInd_v;
%             gInd_v = gInd_v(:)';
%             
%             gInd_u = reshape(gInd_u,model.k,model.q);
%             %gInd = gInd' * model.dynamics.Kt;
%             gInd_u =  model.dynamics.Kt * gInd_u;
%             gInd_u = gInd_u(:)';
%         end
%         %gVarmeans(model.inducingIndices, :) = gVarmeans(model.inducingIndices,
%         %:) + gInd; % This should work AFTER reshaping the matrices...but here
%         %we use all the indices anyway.
%         gVarmeans = gVarmeans + gInd_v + gInd_u;
%         gInd_v = []; % Inducing points are not free variables anymore, they dont have derivatives on their own.
%         gInd_u = [];
%     end
    gInd_u = [];

    gVar = [gVarmeans gVarcovs];
    
    if isfield(model.vardist,'paramGroups')
        gVar = gVar*model.vardist.paramGroups;
    end
    
    
    % If we only want to exclude the derivatives for the variational
    % distribution, the following big block will be skipped.
    if ~(isfield(model, 'onlyKernel') && model.onlyKernel)
        % It may better to start optimize beta a bit later so that
        % so that the rest parameters can be initialized
        % (this could help to escape from the trivial local
        % minima where the noise beta explains all the data)
        % The learnBeta option deals with the above.
        
        % This constrains the variance of the dynamics kernel to one
        % (This piece of code needs to be done in better way with unit variance dynamic
        %  kernels. The code below also will only work for rbf dynamic kernel)
        % Assume that the base rbf/matern/etc kernel is first in the compound
        % structure
        if isfield(model, 'dynamics') && ~isempty(model.dynamics)
            if strcmp(model.dynamics.kern.comp{1}.type,'rbf') || strcmp(model.dynamics.kern.comp{1}.type,'matern32') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic2')
                if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
                    gDynKern(2) = 0;
                end
            end
            
            %___NEW: assume that the second rbf/matern etc kernel is last in the
            %compound kernel
            %if numel(model.dynamics.kern.comp) > 3
            if isfield(model.dynamics, 'learnSecondVariance') && ~model.dynamics.learnSecondVariance   %%%%% NEW
                gDynKern(end) = 0;
            end
            %end
            %___
        end
    end
    
    if isempty(gVar)
        gVar = zeros(1,model.q*model.N*2);
    end
    if isempty(gInd_v)
        gInd_v = 0*model.X_v(:)';
    end
    if isempty(gInd_u)
        gInd_u = 0*model.X_u(:)';
    end
    if isempty(gKern_v)
        gKern_v = zeros(1,model.kern_v.nParams);
    end
    if isempty(gKern_u)
        gKern_u = zeros(1,model.kern_u.nParams);
    end
    if isempty(gDynKern)
        gDynKern= zeros(1,model.dynamics.kern.nParams);
    end
    %%
    % In case we are in the phase where the vardistr. is initialised (see above
    % for the variance of the kernel), beta is kept fixed. For backwards
    % compatibility this can be controlled either with the learnBeta field or
    % with the initVardist field. The later overrides the first.
    
     if isfield(model, 'learnBeta') && model.learnBeta
        gBetaFinal = gBeta;
    else
        gBetaFinal = 0*gBeta;
     end
    
    if isfield(modelAll, 'onlyOptiVardist') && modelAll.onlyOptiVardist == 1 
        gDynKern = 0*gDynKern;
        gBetaFinal = 0*gBetaFinal;
        gW = 0*gW;
        gKern_v = 0*gKern_v;
        gKern_u = 0*gKern_u;
        galpha = 0*galpha;
        gepsilon = 0*gepsilon;
    end
    if isfield(modelAll, 'onlyOptiModel') && modelAll.onlyOptiModel == 1 
        gVar = 0*gVar;
        gInd_v = 0*gInd_v;
        gInd_u = 0*gInd_u;
    end
    
    if isfield(modelAll, 'fixW') && modelAll.fixW == 1 
        gW = 0*gW;
    end
    if isfield(modelAll, 'fixEpsilon') && modelAll.fixEpsilon == 1 
        gepsilon = 0*gepsilon;
    end
    if isfield(modelAll, 'fixDynKern') && modelAll.fixDynKern == 1 
        gDynKern = 0*gDynKern;
    end
    if isfield(modelAll, 'fixBeta') && modelAll.fixBeta == 1 
        gBetaFinal = 0*gBetaFinal;
    end
    if isfield(modelAll, 'fixAlpha') && modelAll.fixAlpha == 1 
        galphaParam = 0*galphaParam;
    end
    
    g = [gVar gDynKern gInd_v gInd_u gBetaFinal gW(:)' gKern_v gKern_u galphaParam gepsilon];
    
    % At this point, gDynKern will be [] if there are no dynamics]
    gPrivAll = [gPrivAll g];
    
    
%     if strcmp(model.kern.type, 'rbfardjit')
%         % different derivatives for the variance, which is super-numerically stable for
%         % this particular kernel
%         if model.learnSigmaf == 1
%             gKern(1) = 0.5*model.d*( - model.k+ sum(sum(model.invLat.*model.invLat))/model.beta - model.beta*(model.Psi0-model.TrC)  )...
%                 + 0.5*tmpV;
%             
%             if ~isstruct(model.kern.transforms(1))
%                 fhandle = str2func([model.kern.transform(1) 'Transform']);
%                 gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%             else
%                 fhandle = str2func([model.kern.transforms(1).type 'Transform']);
%                 if ~isfield(model.kern.transforms(1), 'transformsettings')
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact');
%                 else
%                     gKern(1) = gKern(1).*fhandle(model.kern.variance, 'gradfact', model.kern.transforms(1).transformsettings);
%                 end
%             end
%         else
%             gKern(1) = 0;
%         end
%     end
    
    
end

gShareDynKernCoeff = 0;
for q=1:model.q
    gShareDynKernCoeff = gShareDynKernCoeff + 0.5*(inv(dynModel.Kt) - inv(dynModel.Kt)*(dynModel.vardist.means(:,q)...
        *dynModel.vardist.means(:,q)' + dynModel.vardist.Sq{q})*inv(dynModel.Kt));
end
gShareDynKern = kernGradient(dynModel.kern, dynModel.t, gShareDynKernCoeff);
gShareDynKern = gShareDynKern;


%%%% ...... was commented...
if isfield(model, 'dynamics') && ~isempty(model.dynamics)
    if strcmp(model.dynamics.kern.comp{1}.type,'rbf') || strcmp(model.dynamics.kern.comp{1}.type,'matern32') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic') || strcmp(model.dynamics.kern.comp{1}.type,'rbfperiodic2')
        if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
            gShareDynKern(2) = 0;
        end
    end
    
    %___NEW: assume that the second rbf/matern etc kernel is last in the
    %compound kernel
    %if numel(model.dynamics.kern.comp) > 3
    if isfield(model.dynamics, 'learnSecondVariance') && ~model.dynamics.learnSecondVariance   %%%%% NEW
        gShareDynKern(end) = 0;
    end
    %end
    %___
end


%---- NEW 2012: This is to fix selectively some of the kernel's parameters
if ~isfield(model.dynamics, 'learnVariance') || ~model.dynamics.learnVariance
    % The field model.dynamics.fixedVariance must have the
    % indexes of the gradient vector that correspond to
    % variances of kernels of the matern class and we wish to
    % not learn them.
    if isfield(model.dynamics, 'fixedKernVariance') && ~isempty(model.dynamics.fixedKernVariance)
        gShareDynKern(model.dynamics.fixedKernVariance) = 0;
    end
end
%----

if isfield(modelAll, 'fixShareDynKern') && modelAll.fixShareDynKern == 1 
        gShareDynKern = 0*gShareDynKern;
end
    
g = [gSharedVar gShareDynKern gPrivAll];

end




function [gK_vv , gK_uu, gPsi0, gPsi1, gPsi2,gPsi3, gPsi4, gPsi5, g_Lambda, gBeta, gW ,gepsilon, tmpV_v ,tmpV_u] = vargpCovGrads(model ,dynModel, modelAll, samd)
    sigm = 1/model.beta; % beta^-1
    lend = length(samd);

    tempMatrix = calculateMatrix( model,samd);

    %with svi
    gBeta_v = 0;

    gPsi0 = model.beta * model.m * model.B_v'; %Npart*M
    gPsi0 = gPsi0'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...M*N
    gPsi2 = -0.5 * model.beta * model.d; %1*1
    gPsi4 = (model.beta/2) * model.T1_v; %30*30s
    gK_vv = 0.5 * (model.T1_v - (model.beta * model.d) * model.invLmT_v * model.C_v * model.invLm_v);%30*30
    PLm_v = model.invLatT_v*model.P_v; %30*59
    tmpV_v = sum(sum(PLm_v.*PLm_v)); %1*1
    gBeta_v = 0.5*(model.d*(model.TrC_v + (model.N-model.k)*sigm -model.Psi2 ) ...
        - model.TrYY + model.TrPP_v + model.d/lend*tempMatrix.TrPP_u  ...
        + (1/model.beta^2 * model.d * sum(sum(model.invLat_v.*model.invLat_v)))...
        + sigm*tmpV_v);

    %     gPsi0 = model.beta * model.m(:,samd) * tempMatrix.B_v'; %Npart*M
    %     gPsi0 = gPsi0'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...M*N
    %     gPsi2 = -0.5 * model.beta * lend; %1*1
    %     gPsi4 = (model.beta/2) * tempMatrix.T1_v; %30*30s
    %     gK_vv = 0.5 * (tempMatrix.T1_v - (model.beta * lend) * model.invLmT_v * tempMatrix.C_v * model.invLm_v);%30*30
    %     PLm_v = tempMatrix.invLatT_v*tempMatrix.P_v; %30*59
    %     tmpV_v = sum(sum(PLm_v.*PLm_v)); %1*1
    %     gBeta_v = 0.5*(lend*(tempMatrix.TrC_v + (model.N-model.k)*sigm -model.Psi2 ) ...
    %         - tempMatrix.TrYYdr + tempMatrix.TrPP_v + tempMatrix.TrPP_u  ...
    %         + (1/model.beta^2 * lend * sum(sum(tempMatrix.invLat_v.*tempMatrix.invLat_v)))...
    %         + sigm*tmpV_v);
    %     gPsi0 = model.d/lend * gPsi0;
    %     gPsi2 = model.d/lend * gPsi2;
    %     gPsi4 = model.d/lend * gPsi4;
    %     gK_vv = model.d/lend * gK_vv;
    %     gBeta_v = model.d/lend * gBeta_v;



    gPsi1_temp = zeros(model.N,model.k);
    gPsi1 = gPsi1_temp';
    gPsi3 = 0;
    gPsi5 = zeros(model.k,model.k);
    gK_uu = zeros(model.k,model.k);
    gW = zeros(model.d,model.J);
    gBeta_u = 0;
    tmpV_u = 0;
    if ~(isfield(model,'fixU') && model.fixU == 1)
        for d = 1:lend
            dd = samd(d);
            for j = 1:model.J

                gPsi1_temp = gPsi1_temp + model.beta * model.W(dd,j)^2 * model.m(:,dd) * tempMatrix.B_u{d,j}';%N*M

                gPsi3 = gPsi3 - 0.5 * model.beta * model.W(dd,j)^2;

                Tb_u = (1/model.beta) * (tempMatrix.P1_u{d,j}' * tempMatrix.P1_u{d,j});
                Tb_u = Tb_u + (model.W(dd,j)^2 * tempMatrix.B_u{d,j} * tempMatrix.B_u{d,j}');
                %         gK_uu = gK_uu + 0.5 * (model.invK_uu - Tb_u - (model.beta) * model.invLmT_u * model.C_u{d,j} * model.invLm_u);%30*30
                gK_uu = gK_uu + 0.5 * (model.invK_uu - Tb_u - (model.beta) * model.invLmT_u * tempMatrix.C_u{d,j} * model.invLm_u);%30*30

                PLm_u = tempMatrix.invLatT_u{d,j}*tempMatrix.P_u{d,j}; %30*59
                tmpV_u = model.W(dd,j)^2 * sum(sum(PLm_u.*PLm_u)); %1*1

                gBeta_u = gBeta_u + 0.5*( tempMatrix.TrC_u(d,j) + (-model.k)*sigm - model.W(dd,j)^2*model.Psi3 ...
                    + (1/model.beta^2 * (sum(sum(tempMatrix.invLat_u{d,j}.*tempMatrix.invLat_u{d,j}))))...
                    + sigm*tmpV_u);

%                 gW(dd,j) = -trace(model.W(dd,j)*tempMatrix.P1_u{d,j}'*tempMatrix.P1_u{d,j}*model.Psi5) ...
%                     + model.beta*model.W(dd,j)*model.m(:,dd)'*model.Psi1*tempMatrix.B_u{d,j}...
%                     -model.beta*model.W(dd,j)^3*trace(tempMatrix.B_u{d,j}*tempMatrix.B_u{d,j}'*model.Psi5)...
%                     -model.beta*model.W(dd,j)*model.Psi3 + model.beta*trace(model.W(dd,j)*model.Psi5*model.invK_uu);
            end
        end

        gPsi5 = (model.beta/2) * tempMatrix.T1_u; %30*30

        % gPsi1 = model.beta * model.W^2 * model.m * model.B_u'; %30*359
        gPsi1 = model.d/lend*gPsi1_temp'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...
        gPsi3 = model.d/lend*gPsi3;
        gPsi5 = model.d/lend * gPsi5;
        gK_uu = model.d/lend*gK_uu;
        gBeta_u = model.d/lend*gBeta_u;
        % gPsi3 = -0.5 * model.beta * model.W^2 * model.d; %1*1
    end

    gBeta = gBeta_v + gBeta_u;

    gepsilon = 0;
    if ~(isfield(model,'fixEpsilon') && model.fixEpsilon == 1)
        for q=1:model.q
            gepsilon = gepsilon - 0.5*trace((inv(model.dynamics.At)- inv(model.dynamics.At)* ...
                ((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))*...
                ((1-model.alpha)*dynModel.vardist.means(:,q)-model.dynamics.vardist.means(:,q))'...
                *inv(model.dynamics.At) - (1-model.alpha)^2*inv(model.dynamics.At)*dynModel.vardist.Sq{q}*inv(model.dynamics.At)...
                - inv(model.dynamics.At)*model.dynamics.vardist.Sq{q}*inv(model.dynamics.At))*(1-model.alpha)^2*eye(model.N));
        end
        gepsilon = gepsilon * expTransform(model.epsilon, 'gradfact');
    end


    %gBeta = 0.5*(model.d*(model.TrC + (model.N-model.k)*sigm -model.Psi0) ...
    %	- model.TrYY + model.TrPP ...
    %	+ sigm * sum(sum(model.K_uu .* model.Tb)));

    if ~isstruct(model.betaTransform)
        fhandle = str2func([model.betaTransform 'Transform']);
        gBeta = gBeta*fhandle(model.beta, 'gradfact');
    else
        fhandle = str2func([model.betaTransform.type 'Transform']);
        gBeta = gBeta*fhandle(model.beta, 'gradfact', model.betaTransform.transformsettings);
    end

    g_Lambda = repmat(-0.5*model.beta*model.d, 1, model.N);

    clear tempMatrix;
end

function g = g1(model)
    % THE FOLLOWING WORKS ONLY WHEN THERE ARE NO DYNAMICS... %___ TODO TEMP
    % Shared params
    if isfield(model, 'dynamics') && isempty(model.dynamics) % TEMP !!!! TODO (~isempty?)
        error('The gradients for the dynamics case are not implemented correctly yet!'); % TEMP
    else % ....
        gVarmeansKL = - model.vardist.means(:)';
        gVarcovsKL = 0.5 - 0.5*model.vardist.covars(:)';
    end
    model = svargplvmPropagateField(model, 'onlyLikelihood', 1);

    %fprintf('# Derivs for KL is (should be zero): ');%%%TEMP

    % Private params
    % g = [[sum_m(gVar_only_likelihood)+gVar_onlyKL] g_1 g_2 ...]
    % where sum_m is the sum over all models and g_m is the gradients for the
    % non-shared parameters for model m
    g = [];
    gShared = [gVarmeansKL gVarcovsKL];

    for i=1:model.numModels
        g_i = vargplvmLogLikeGradients(model.comp{i});
        % Now add the derivatives for the shared parameters.
        if isfield(model, 'dynamics') & ~isempty(model.dynamics) % !! (doesn't work correctly for dynamics)
            gShared = gShared + g_i(1:model.dynamics.nParams); % !! (doesn't work correctly for dynamics)
            g_i = g_i((model.dynamics.nParams+1):end); % TEMP (doesn't work correctly for dynamics)
        else % else it's only the vardist. of the KL
            gShared = gShared + g_i(1:model.vardist.nParams);
            g_i = g_i((model.vardist.nParams+1):end);
        end
        g = [g g_i];
    end
    g = [gShared g];
end

% NOT TESTED FOR THE DYNAMICS CASE
function g = g2(model)
    g_1 = vargplvmLogLikeGradients(model.comp{1});
    gShared = g_1(1:model.vardist.nParams);
    g_1 = g_1(model.vardist.nParams+1:end);
    model = svargplvmPropagateField(model, 'onlyLikelihood', 1, true);
    g=[];
    for i=2:model.numModels
        g_i = vargplvmLogLikeGradients(model.comp{i});
        % Now add the derivatives for the shared parameters.
        if isfield(model, 'dynamics') & ~isempty(model.dynamics)
            gShared = gShared + g_i(1:model.dynamics.nParams);
            g_i = g_i((model.dynamics.nParams+1):end);
        else % else it's only the vardist. of the KL
            gShared = gShared + g_i(1:model.vardist.nParams);
            g_i = g_i((model.vardist.nParams+1):end);
        end
        g = [g g_i];
    end
    g = [gShared g_1 g];
end


function g = gPar(model)

    % THE FOLLOWING WORKS ONLY WHEN THERE ARE NO DYNAMICS... %___ TODO TEMP
    % Shared params
    if isfield(model, 'dynamics') && isempty(model.dynamics) % TEMP
        error('The gradients for the dynamics case are not implemented correctly yet!'); % TEMP
    else % ....
        gVarmeansKL = - model.vardist.means(:)';
        gVarcovsKL = 0.5 - 0.5*model.vardist.covars(:)';
    end
    model = svargplvmPropagateField(model, 'onlyLikelihood', 1);

    %fprintf('# Derivs for KL is (should be zero): ');%%%TEMP

    % Private params
    % g = [[sum_m(gVar_only_likelihood)+gVar_onlyKL] g_1 g_2 ...]
    % where sum_m is the sum over all models and g_m is the gradients for the
    % non-shared parameters for model m
    gShared = [gVarmeansKL gVarcovsKL];
    modelTemp=model.comp;
    parfor i=1:model.numModels
        gAll{i} = vargplvmLogLikeGradients(modelTemp{i});
    end
    g=[];
    for i=1:model.numModels
        g_i = gAll{i};
        % Now add the derivatives for the shared parameters.
        if isfield(model, 'dynamics') & ~isempty(model.dynamics) % TEMP (doesn't work correctly for dynamics)
            gShared = gShared + g_i(1:model.dynamics.nParams); % TEMP (doesn't work correctly for dynamics)
            g_i = g_i((model.dynamics.nParams+1):end); % TEMP (doesn't work correctly for dynamics)
        else % else it's only the vardist. of the KL
            gShared = gShared + g_i(1:model.vardist.nParams);
            g_i = g_i((model.vardist.nParams+1):end);
        end
        g = [g g_i];
    end
    g = [gShared g];

end

function ll = mvcgpdsLogLikelihood(param, model, samd)

% mvcgpdsLogLikelihood Log-likelihood for a mvcgpds.
% FORMAT
% DESC returns the log likelihood for a given mvcgpds model.
% ARG model : the model for which the log likelihood is to be
% computed. The model contains the data for which the likelihood is
% being computed in the 'y' component of the structure.
% RETURN ll : the log likelihood of the data given the model.
%
% SEEALSO:  mvcgpdsLogLikeGradients


% Functions f1 and f2 should be equivalent in the static case, but f1 might
% (?) be faster.
% if ~isfield(model, 'dynamics') || isempty(model.dynamics)
%     ll = f1(model, samd);
%     %ll = f2(model);
% else
    ll = f2(model, samd);
    for i = 1:model.numModels
        if isfield(model.comp{i}, 'KLweight')
            assert(model.comp{i}.KLweight == 0.5); % not implemented yet for dynamics
        end
    end
% end

% Here we take advantage of the fact that in the bound, the likelihood and
% the KL part break into ll+KL. So, in the shared case, we only have
% KL + ll1 + ll2 + ...
function lower_bound = f2(model, samd)

% We want to count for the KL part only once, because it is shared. Then, we
% will add all likelihood parts from all sub-models. The above can be
% achieved by calculating the bound for the first sub-model as usual, and
% then calculate only the likelihood part for the rest of the sub-models
% (because the KL part would be the same for all models since we called
% expandParam previously).

%make onlyLikelihood is true.
model = svargplvmPropagateField(model, 'onlyLikelihood', 1, true);

ll=0;
for i=1:model.numModels
    samd = [1:model.comp{i}.d];
    ll = ll + cgpdsLogLikelihood(model.comp{i}, samd); % ll = ll + ll_i
end

KLdiv = 0;
Kt_share = model.dynamics.Kt;
mu_share = model.dynamics.vardist.means;
S_share = model.dynamics.vardist.Sq;
for q=1:model.q
    if(det(Kt_share)==0)
%         KLdiv = KLdiv - 1e5; 
    else
        KLdiv = KLdiv + log(det(Kt_share));
    end
    if(det(S_share{q})==0)
%         KLdiv = KLdiv - 1e5;
    else
        KLdiv = KLdiv - log(det(S_share{q}));
    end

%     if(det(Kt_share)==0) %if the value of the det is equal to zero, the log(det(...)) will be -INF, here we use -1e5 to approximate -INF.
%         Kt_share = positiveDefiniteMatrix(Kt_share);
%     end
%     KLdiv = KLdiv + log(det(Kt_share));
%     
%     if(det(S_share{q})==0)
%         S_share{q} = positiveDefiniteMatrix( S_share{q} );
%     end
%     KLdiv = KLdiv - log(det(S_share{q}));

    
    KLdiv = KLdiv + trace(inv(Kt_share)*(mu_share(:,q)*mu_share(:,q)'+ S_share{q}));
end


for i = 1:model.numModels
    At = model.comp{i}.dynamics.At;
    mu = model.comp{i}.dynamics.vardist.means;
    alpha = model.comp{i}.alpha;
    S = model.comp{i}.dynamics.vardist.Sq;
    for q=1:model.q
        if(det(At)==0)
%             KLdiv = KLdiv - 1e5;
        else
            KLdiv = KLdiv + log(det(At));
        end
        if(det(S{q})==0)
%             KLdiv = KLdiv - 1e5;
        else
            KLdiv = KLdiv - log(det(S{q}));
        end

%         if(det(At)==0)
%             At = positiveDefiniteMatrix( At );
%         end
%         KLdiv = KLdiv + log(det(At));
%         
%         if(det(S{q})==0)
%            S{q} = positiveDefiniteMatrix( S{q} );
%         end
%         KLdiv = KLdiv - log(det(S{q}));
         
        KLdiv = KLdiv + trace(inv(At)*S{q})+ trace(((1-alpha)^2*inv(At))*(S_share{q}))...
            + ((1-alpha)*mu_share(:,q) - mu(:,q))'*inv(At)*((1-alpha)*mu_share(:,q) - mu(:,q));

    end
end


KLdiv = 0.5*KLdiv;

lower_bound =  ll - KLdiv;


%f1 is work for non-dynamical situations.
function f = f1(model, samd)

% model.numModels=1; %%%%%%%%TEMP

% This works only when there are NOT any dynamics:
varmeans = sum(sum(model.vardist.means.*model.vardist.means));
varcovs = sum(sum(model.vardist.covars - log(model.vardist.covars)));
KLdiv = -0.5*(varmeans + varcovs) + 0.5*model.q*model.N;

model = svargplvmPropagateField(model, 'onlyLikelihood', 1);
ll=0;
for i=1:model.numModels
    samd = [1:model.comp{i}.d];
    ll = ll + vargplvmLogLikelihood(model.comp{i}, samd);
end
f = (ll + KLdiv);



function model = mvcgpdsCreate(q, d, J, Y, options)

% mvcgpdsCreate Create a mvcgpds model with inducing variables.
%
%	Description:
%
%	MODEL = mvcgpdsCreate(Q, D, J, Y, OPTIONS, ENABLEDGTN) creates a
%	mvcgpds model with the possibility of using inducing variables to
%	speed up computation.
%	 Returns:
%	  MODEL - the mvcgpds model.
%	 Arguments:
%	  Q - dimensionality of latent space.
%	  D - dimensionality of data space.
%     J - the number of latent g
%	  Y - the data to be modelled in design matrix format (as many rows
%	   as there are data points).
%	  OPTIONS - options structure as returned from VARGPLVMOPTIONS. This
%	   structure determines the type of approximations to be used (if
%	   any).


% if size(Y, 2) ~= d
%     error(['Input matrix Y does not have dimension ' num2str(d)]);
% end


if isfield(options,'enableDgtN')
	enableDgtN = options.enableDgtN;
else
	enableDgtN = true; % default behaviour is to enable D >> N mode
end

% Datasets with dimensions larger than this number will run the
% code which is more efficient for large D. This will happen only if N is
% smaller than D, though.
if enableDgtN
    limitDimensions = 5000; % Default: 5000
else
    limitDimensions = 1e+10;
end

model.type = 'cgpds';
model.approx = options.approx;

model.learnScales = options.learnScales;
%model.scaleTransform = optimiDefaultConstraint('positive');

model.optimiseBeta = options.optimiseBeta;

if isfield(options, 'betaTransform') && isstruct( options.betaTransform )
    model.betaTransform = options.betaTransform;
else
    model.betaTransform =  optimiDefaultConstraint('positive');
end

model.q = q;
model.d = d;
model.N = size(Y, 1);
model.J = J;

model.optimiser = options.optimiser;
model.bias = mean(Y);
model.scale = ones(1, model.d);

if(isfield(options,'scale2var1'))
    if(options.scale2var1)
        fprintf('# Scaling output data to variance one...\n')
        model.scale = std(Y);
        model.scale(find(model.scale==0)) = 1;
        if(model.learnScales)
            warning('Both learn scales and scale2var1 set for GP');
        end
        if(isfield(options, 'scaleVal'))
            warning('Both scale2var1 and scaleVal set for GP');
        end
    end
end

if(isfield(options, 'scaleVal'))
    if(length(options.scaleVal)==1)
        model.scale = repmat(options.scaleVal, 1, model.d);
    elseif(length(options.scaleVal)==model.d)
        model.scale = options.scaleVal;
    else
        error('Dimension mismatch in scaleVal');
    end
end


model.y = Y;
model.m = gpComputeM(model);%对数据进行归一化，归成标准高斯N(0,I)

% Marg. likelihood is a sum ML = L + KL. Here we allow for
% ML = 2*((1-fw)*L + fw*KL), fw = options.KLweight. Default is 0.5, giving
% same weight to both terms.
model.KLweight = options.KLweight;
assert(model.KLweight >=0 && model.KLweight <=1);


if isstr(options.init_X_share)
    %%% The following should eventually be uncommented so as to initialize
    %%% in the dual space. This is much more efficient for large D.
    %if model.d > limitDimensions && model.N < limitDimensions
    %      [X vals] = svd(model.m*model.m');
    %      X = X(:,1:model.q); %%%%%
    %else
    initFunc = str2func([options.initX 'Embed']);
    X = initFunc(model.m, q);
    %end
else
    if size(options.init_X_share, 1) == size(Y, 1) ...
            & size(options.init_X_share, 2) == q
        X = options.init_X_share;
    else
        error('options.initX not in recognisable form.');
    end
end

model.X = X;
model.learnBeta = 1;

% If the provided matrices are really big, the required quantities can be
% computed externally (e.g. block by block) and be provided here (we still
% need to add a switch to get that).
if model.d > limitDimensions && model.N < limitDimensions
    model.DgtN = 1; % D greater than N mode on.
    fprintf(1, '# The dataset has a large number of dimensions (%d)! Switching to "large D" mode!\n',model.d);
    
    % Keep the original m. It is needed for predictions.
    model.mOrig = model.m;
    
    % The following will eventually be uncommented.
    %model.y = []; % Never used.
    YYT = model.m * model.m'; % NxN
    % Replace data with the cholesky of Y*Y'.Same effect, since Y only appears as Y*Y'.
    %%% model.m = chol(YYT, 'lower');  %%% Put a switch here!!!!
    [U S V]=svd(YYT);
    model.m=U*sqrt(abs(S));
    
    model.TrYY = sum(diag(YYT)); % scalar
else
    % Trace(Y*Y) is a constant, so it can be calculated just once and stored
    % in memory to be used whenever needed.
    model.DgtN = 0;
    model.TrYY = sum(sum(model.m .* model.m));
end

model.date = date;

%%% Also, if there is a test dataset, Yts, then when we need to take
% model.m*my' (my=Yts_m(i,:)) in the pointLogLikeGradient, we can prepare in advance
% A = model.m*Yts_m' (NxN) and select A(:,i).
% Similarly, when I have m_new = [my;m] and then m_new*m_new', I can find that
% by taking m*m' (already computed as YY) and add one row on top: (m*my)'
% and one column on the left: m*my.
%
% %%%%%%%%%%

%%%% _NEW



%model.isMissingData = options.isMissingData;
%if model.isMissingData
%  for i = 1:model.d
%    model.indexPresent{i} = find(~isnan(y(:, i)));
% end
%end
%model.isSpherical = options.isSpherical;


if isstruct(options.kern)
    model.kern_u = options.kern;
    model.kern_v = options.kern;
else
    model.kern_u = kernCreate(model.X, options.kern,d);%%%%%%the same kern??????
    model.kern_v = kernCreate(model.X, options.kern,d);
end


% check if parameters are to be optimised in model space (and constraints are handled
% by optimiser)
if isfield(options, 'notransform') && options.notransform == true
    % store notransform option in model:
    % this is needed when computing gradients
    model.notransform = true;
    model.vardist = vardistCreate(X, q, 'gaussian', 'identity');
else
    model.vardist = vardistCreate(X, q, 'gaussian'); %%%
end

switch options.approx
    case {'dtcvar'}
        % Sub-sample inducing variables.
        model.k = options.numActive;
        model.fixInducing = options.fixInducing;
        if options.fixInducing
            if length(options.fixIndices)~=options.numActive
                error(['Length of indices for fixed inducing variables must ' ...
                    'match number of inducing variables']);
            end
            model.X_u = model.X(options.fixIndices, :);
            model.X_v = model.X(options.fixIndices, :);
            model.inducingIndices = options.fixIndices;
        else
            %%%NEW_: make it work even if k>N
            if model.k <= model.N
                if ~isfield(options, 'labels')
                    ind = randperm(model.N);
                    ind_v = ind(1:model.k);%the first model.k
%                     ind_u = ind(model.k+1:2*model.k);
                    ind_u = ind(model.N-model.k+1:model.N);%the last model.k
                    model.X_v = model.X(ind_v, :);
                    model.X_u = model.X(ind_u, :);
                else
                    % in the case that class labels are supplied, make sure that inducing inputs
                    % from all classes are chosen
                    [idcs, nSmpls] = class_samples( options.labels, model.k );
                    
                    count = 1;
                    midx = [];
                    for inds = idcs
                        ind   = inds{:};
                        ind   = ind(randperm(numel(ind)));                            
                        idx  = ind(1:nSmpls(count));
                        
                        % test that there is no overlap between index sets
                        assert(isempty(intersect(midx, idx)));
                        midx = [midx, idx];
                        
                        count = count+1;
                    end    
                    model.X_u = model.X(midx,:);
                    model.X_v = model.X(midx,:);
                end                    
            else
                % TODO: sample from the variational distr. (this should probably go
                % to the dynamics as well because the vardist. changes in the initialization for the dynamics.

                %!!! The following code needs some more testing!
                samplingInd=0; %% TEMP
                if samplingInd
                    % This only works if k<= 2*N
                    model.X_u=zeros(model.k, model.q);
                    ind = randperm(model.N);
                    %ind = ind(1:model.N);
                    model.X_u(1:model.N,:) = model.X(ind, :);

                    % The remaining k-N points are sampled from the (k-N) first
                    % distributions of the variational distribution (this could be done
                    % randomly as well).
                    dif=model.k-model.N;
                    model.X_u(model.N+1:model.N+dif,:)= model.vardist.means(1:dif,:) + rand(size(model.vardist.means(1:dif,:))).*sqrt(model.vardist.covars(1:dif,:));  % Sampling from a Gaussian.
                    model.X_v = model.X_u;
                else
                    model.X_u=zeros(model.k, model.q);
                    for i=1:model.k
                        %ind=randi([1 size(model.vardist.means,1)]);
                        % Some versions do not have randi... do it with rendperm
                        % instead: 
                        % ceil(size(model.vardist.means,1).*rand) % alternative
                        ind=randperm(size(model.vardist.means,1));
                        ind=ind(1);
                        model.X_u(i,:) = model.vardist.means(ind,:);
                        model.X_v = model.X_u;
                    end
                end
            end
            %%%_NEW
        end
        
        model.beta = options.beta;
        model.W = 1e-6*rand(model.d,model.J);
        model.alpha = 0.5;
        model.epsilon = 1/options.beta;
end
%{
%if model.k>model.N
%  error('Number of active points cannot be greater than number of data.')
%end
%if strcmp(model.approx, 'pitc')
%  numBlocks = ceil(model.N/model.k);
%  numPerBlock = ceil(model.N/numBlocks);
%  startVal = 1;
%  endVal = model.k;
%  model.blockEnd = zeros(1, numBlocks);
%  for i = 1:numBlocks
%    model.blockEnd(i) = endVal;
%    endVal = numPerBlock + endVal;
%    if endVal>model.N
%      endVal = model.N;
%    end
%  end
%end
%}

if isstruct(options.prior)
    model.prior = options.prior;
else
    if ~isempty(options.prior)
        model.prior = priorCreate(options.prior);
    end
end

if isfield(options, 'notransform') && options.notransform == true
    model.prior.transforms.type = 'identity';    
end

%model.vardist = vardistCreate(X, q, 'gaussian');

if isfield(options, 'tieParam') & ~isempty(options.tieParam)
    %
    if strcmp(options.tieParam,'free')
        % paramsList =
    else
        startVal = model.vardist.latentDimension*model.vardist.numData + 1;
        endVal = model.vardist.latentDimension*model.vardist.numData;
        for q=1:model.vardist.latentDimension
            endVal = endVal + model.vardist.numData;
            index = startVal:endVal;
            paramsList{q} = index;
            startVal = endVal + 1;
        end
        model.vardist = modelTieParam(model.vardist, paramsList);
    end
    %
end

%{
%  if isstruct(options.back)

%if isstruct(options.inducingPrior)
%  model.inducingPrior = options.inducingPrior;
%else
%  if ~isempty(options.inducingPrior)
%    model.inducingPrior = priorCreate(options.inducingPrior);
%  end
%end

%if isfield(options, 'back') & ~isempty(options.back)
%  if isstruct(options.back)
%    model.back = options.back;
%  else
%    if ~isempty(options.back)
%      model.back = modelCreate(options.back, model.d, model.q, options.backOptions);
%    end
%  end
%  if options.optimiseInitBack
%    % Match back model to initialisation.
%    model.back = mappingOptimise(model.back, model.y, model.X);
%  end
%  % Now update latent positions with the back constraints output.
%  model.X = modelOut(model.back, model.y);
%else
%  model.back = [];
%end

model.constraints = {};

model.dynamics = [];

initParams = vargplvmExtractParam(model);
model.numParams = length(initParams);

% This forces kernel computation.
model = vargplvmExpandParam(model, initParams);
%}

end


function [idcs, nSmpls] = class_samples( lbls, nActive )

    if size(lbls,1) ~= 1
        lbls = lbls';
    end
    
    clids  = unique(lbls);
    nEx    = zeros(1, numel(clids));
    nSmpls = zeros(1, numel(clids));
    idcs   = cell( 1, numel(clids));
    
    count = 1;
    for c = clids 
        tmp = find(lbls==c);
        idcs{count} = tmp;
        nEx(count)  = numel(tmp);
        nSmpTmp =  ceil(nEx(count)/length(lbls)*nActive);
        if nSmpTmp > nEx(count)
            nSmpTmp = nEx(count);
        end
        nSmpls(count) = nSmpTmp;
       
        count = count + 1;
    end
    
    if sum(nEx) < nActive
       error('There must be at least as many training samples as inducing variables.');
    end
    
    nRes = nActive - sum(nSmpls);
    if nRes > 0
        while nRes > 0
            [nSmplsSort, ind] = sort(nSmpls,'ascend');
            for count = ind
                if nSmpls(count) < nEx(count)
                    nSmpls(count) = nSmpls(count)+1;
                    nRes = nRes-1;
                end

                if nRes == 0
                    break;
                end
            end
        end
    else
       while nRes < 0
          [mx, ind] = max(nSmpls);
          nSmpls(ind) = nSmpls(ind) - 1;
          nRes = nRes + 1;
       end
    end
end



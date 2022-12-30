% DEMHUMANPOSEMVCGPDS1 Run the multi-view CGPDS on the humanEva dataset.
%

%___
clear ;close all;
% Fix seeds
randn('seed', 1e5);
rand('seed', 1e5);

% Define constants (in a manner that allows other scripts to parametrize
% this one).
if ~exist('experimentNo') ,  experimentNo = 4;      end
if ~exist('isTraining') ,  isTraining = 1;      end
if ~exist('isOptimizingInPrediction') ,  isOptimizingInPrediction = 1;      end
if ~exist('indPoints')    ,  indPoints = 100;          end     % Default: 49
if ~exist('latentDimPerModel')    ,  latentDimPerModel = 5;          end
% Set to 1 to use dynamics
if ~exist('dynUsed')      ,  dynUsed = 1;             end
if ~exist('dynamicKern')   ,     dynamicKern = {'rbf', 'white', 'bias'}; end
if ~exist('mappingKern')   ,  mappingKern = {'rbfard2', 'white', 'bias'}; end
if ~exist('dynamicsConstrainType '),  dynamicsConstrainType = {'time'}; end
% 0.1 gives around 0.5 init.covars. 1.3 biases towards 0.
if ~exist('vardistCovarsMult'),  vardistCovarsMult=[]; end %1.3; % EDIT from submission.
% Set to empty value {} to work with toy data

if ~exist('invWidthMultDyn'),    invWidthMultDyn = 100;                     end
if ~exist('invWidthMult'),       invWidthMult = 5;                     end
if ~exist('initX'),     initX ='ppca';   end % That's for the dynamics initialisation
if ~exist('dataType'), dataType = 'humanPose'; end
if ~exist('enableParallelism'), enableParallelism = 1; end
if ~exist('DgtN'), DgtN = false; end
% Create initial X by doing e.g. ppca in the concatenated model.m's or by
% doing ppca in the model.m's separately and concatenate afterwards?
if ~exist('initial_X'), initial_X = 'concatenated'; end % Other options: 'concatenated'
% Which indices to use for training, rest for test
if ~exist('indTr'), indTr = -1; end
if ~exist('usePixels'), usePixels = false; end
if ~exist('parallel', 'var'), parallel = false; end
if ~exist('addBetaPrior','var'), addBetaPrior = false; end
if ~exist('priorScale','var'), priorScale = 1; end
if ~exist('useScaleVal','var'), useScaleVal = true; end

demHumanPosePrepareData

dataSetNames={'silhouette', 'pose'};
% X_init = Xp;
%mappingKern = {'linard2', 'white'};
%mappingKern = {'rbfard2', 'white'};
latentDim = 5; % Anything > 2 and < 10
%xyzankurAnim(Z_test, 3);
numberOfDatasets = length(Yall);

%-- Load datasets
for i=1:numberOfDatasets
    Y = Yall{i};
    
    dims{i} = size(Y,2);
    N{i} = size(Y,1);
    if indTr == -1
        indTr = 1:N{i};
    end
    %t{i} = linspace(0, 2*pi, size(Yall{i}, 1)+1)'; t{i} = t{i}(1:end-1, 1);
    
    indTs = setdiff(1:size(Y,1), indTr);
    Ytr{i} = Y(indTr,:);
    Yts{i} = Y(indTs,:);
    
    d{i} = size(Ytr{i}, 2);
end
% timeStampsTraining = t{1}(indTr,1); %timeStampsTest = t(indTs,1);

t = linspace(0, 2*pi, length(indTr)+1)'; t = t(1:end-1, 1);

% Fix times:
prevSeq = 1;
timeStampsTraining = [];
dt=0.05;
for i=1:length(seq)
    t = ([0:(seq(i)-prevSeq)].*dt)';
    prevSeq = seq(i)+1;
    timeStampsTraining = [timeStampsTraining ;t];
end;

dt=t(2)-t(1);
timeStampsTest = ([0:size(Y_test,1)-1].*dt)';

for i=2:numberOfDatasets
    if N{i} ~= N{i-1}
        error('The number of observations in each dataset must be the same!');
    end
end


%{
for i=1:size(Ytr{1},1)
    %figure
    %subplot(1,2,2)
    %imagesc(reshape(Ytr{1}(i,:), height, width)), colormap('gray');
    %subplot(1,2,1)
    handle = xyzankurVisualise2(Ytr{2}(i,:));
    pause;
    %close
    i
end
xyzankurAnim2(Ytr{2})
%}

%%
svargplvm_init;
%-- Options for the models
for i=1:numberOfDatasets
    % Set up models
    options{i} = vargplvmOptions('dtcvar');
    options{i}.kern = mappingKern; %{'rbfard2', 'bias', 'white'};
    options{i}.numActive = indPoints;
    options{i}.optimiser = 'scg';
    if ~DgtN
        options{i}.enableDgtN = false;
    end
    % !!!!! Be careful to use the same type of scaling and bias for all
    % models!!!
    
    % scale = std(Ytr);
    % scale(find(scale==0)) = 1;
    %options.scaleVal = mean(std(Ytr));
    if useScaleVal
        options{i}.scaleVal = sqrt(var(Ytr{i}(:)));
    end
    if dynUsed
        optionsDyn{i}.type = 'vargpTime';
        optionsDyn{i}.t=timeStampsTraining;
        optionsDyn{i}.inverseWidth=30;
        optionsDyn{i}.seq = seq;
        % Fill in with default values whatever is not already set
        optionsDyn{i} = vargplvmOptionsDyn(optionsDyn{i});
    end
end

%-------------- INIT LATENT SPACE ---%
for i=1:length(Ytr)
    if usePixels && (i ==1)
        m{1} = Ytr{i};
        medTemp = (max(max(m{1})) + min(min(m{1})))/2;
        ind1 = (m{1} > medTemp);
        ind2 = (m{1} <= medTemp);
        m{1}(ind1) = 1;
        m{1}(ind2) = -1;
    else
        % Compute m, the normalised version of Ytr (to be used for
        % initialisation of X)
        bias = mean(Ytr{i});
        scale = ones(1, d{i});
        
        if(isfield(options{i},'scale2var1'))
            if(options{i}.scale2var1)
                scale = std(Ytr{i});
                scale(find(scale==0)) = 1;
                if(isfield(options{i}, 'scaleVal'))
                    warning('Both scale2var1 and scaleVal set for GP');
                end
            end
        end
        if(isfield(options{i}, 'scaleVal'))
            scale = repmat(options{i}.scaleVal, 1, d{i});
        end
        
        % Remove bias and apply scale.
        m{i} = Ytr{i};
        for j = 1:d{i}
            m{i}(:, j) = m{i}(:, j) - bias(j);
            if scale(j)
                m{i}(:, j) = m{i}(:, j)/scale(j);
            end
        end
    end
    %    mAll = [mAll m{i}]; % Concatenation (doesn't work if different sizes)
end

embedFunc = str2func([initX 'Embed']);
X_init_temp = cell(2,1);
if strcmp(initial_X, 'separately')
    fprintf('# Initialising X by performing ppca in each observed (scaled) dataset separately and then concatenating...\n');
    %     X_init{1} = ppcaEmbed(m{1},latentDimPerModel);
    %     X_init{2} = ppcaEmbed(m{2},latentDimPerModel);
    if ~exist('latentDimPerModel') || length(latentDimPerModel)~=2
        latentDimPerModel = [7 3];
    end
    if strcmp(initX, 'vargplvm')
        X_init_temp{1} = embedFunc(m{1},latentDimPerModel(1),[],120,30);
        X_init_temp{2} = embedFunc(m{2},latentDimPerModel(2),[],120,30);
    else
        X_init_temp{1} = embedFunc(m{1},latentDimPerModel(1));
        X_init_temp{2} = embedFunc(m{2},latentDimPerModel(2));
    end
    X_init_share = [X_init_temp{1} X_init_temp{2}];
else
    fprintf('# Initialising X by performing ppca in concatenated observed (scaled) data...\n');
    X_init_share = embedFunc([m{1} m{2}], latentDimPerModel*2);
end

latentDim = size(X_init_share,2);

% Free up some memory
clear('Y')

fileToSave = [pwd filesep 'matlab' filesep 'mvcgpdsResultMat' filesep 'dem' 'HumanPose' 'Mvcgpds' num2str(experimentNo) '.mat'];
LastFile = [pwd filesep 'matlab' filesep 'mvcgpdsResultMat' filesep 'dem' 'HumanPose' 'Mvcgpds' num2str(experimentNo-1) '.mat'];
 
try
    load(LastFile);
catch
    for i=1:numberOfDatasets
        % initial the private latent space with the same value as the share
        % latent sapce
        X_init_private{i} = X_init_share;
        d{i} = size(Ytr{i},2);
        J{i} = 1;
        %---- Here put some code to assign X to the global common X which must
        % be created by doing pca in the concatenation of Y's...After this
        % point, model{i}.X will be the same for all i's. TODO...
        % fprintf(1,'# Creating the model...\n');
        options{i}.init_X_share = X_init_share;
        options{i}.init_X_private = X_init_private{i};
        
        
        model{i} = mvcgpdsCreate(latentDim, d{i}, J{i}, Ytr{i}, options{i});
        
        model{i}.X = X_init_private{i}; %%%%%%%
        samd = [1:model{i}.d];
        model{i} = mvcgpdsParamInit(model{i}, m{i}, model{i}.X, samd);
        %model{i}.X = X_init; %%%%%%%
        var_mi = var(m{i}(:));
        m{i} = []; % Save some memory
        
        inpScales = invWidthMult./(((max(model{i}.X)-min(model{i}.X))).^2); % Default 5
        
        model{i}.kern_v.comp{1}.inputScales = inpScales;
        if ~iscell(model{i}.kern_v)
            model{i}.kern_v.inputScales = model{i}.kern_v.comp{1}.inputScales;
        end
        
        model{i}.kern_u.comp{1}.inputScales = inpScales;
        if ~iscell(model{i}.kern_u)
            model{i}.kern_u.inputScales = model{i}.kern_u.comp{1}.inputScales;
        end
        
        params = cgpdsExtractParam(model{i},samd);
        model{i} = cgpdsExpandParam(model{i}, params, samd);
        model{i}.vardist.covars = 0.5*ones(size(model{i}.vardist.covars)) + 0.001*randn(size(model{i}.vardist.covars));
        
        
        %-------- Add dynamics to the model -----
        if  dynUsed
            fprintf(1,'# Adding dynamics to the model...\n');
            optionsDyn{i}.type = 'vargpTime';
            optionsDyn{i}.t =timeStampsTraining;
            optionsDyn{i}.inverseWidth=invWidthMultDyn; % Default: 100
            optionsDyn{i}.initX = X_init_share;
            kern = kernCreate(optionsDyn{i}.t, dynamicKern); % Default: {'rbf','white','bias'}
            
            %---- Default values for dynamics kernel
            if strcmp(kern.comp{2}.type, 'white')
                kern.comp{2}.variance = 1e-2; % Usual values: 1e-1, 1e-3
            end
            
            if strcmp(kern.comp{2}.type, 'whitefixed')
                if ~exist('whiteVar')
                    whiteVar = 1e-6;
                end
                kern.comp{2}.variance = whiteVar;
                fprintf(1,'# fixedwhite variance: %d\n',whiteVar);
            end
            
            if strcmp(kern.comp{1}.type, 'rbfperiodic')
                if exist('periodicPeriod')
                    kern.comp{1}.period = periodicPeriod;
                end
                fprintf(1,'# periodic period: %d\n',kern.comp{1}.period);
            end
            % The following is related to the expected number of
            % zero-crossings.(larger inv.width numerator, rougher func)
            if ~strcmp(kern.comp{1}.type,'ou')
                kern.comp{1}.inverseWidth = optionsDyn{i}.inverseWidth./(((max(t)-min(t))).^2);
                kern.comp{1}.variance = 1;
            end
            
            optionsDyn{i}.kern = kern;
            optionsDyn{i}.vardistCovars = vardistCovarsMult; % 0.23 gives true vardist.covars around 0.5 (DEFAULT: 0.23) for the ocean dataset
            
            % Fill in with default values whatever is not already set
            optionsDyn{i} = vargplvmOptionsDyn(optionsDyn{i});
            
            model{i} = vargplvmAddDynamics(model{i}, 'vargpTime', samd, optionsDyn{i}, optionsDyn{i}.t, 0, 0,optionsDyn{i}.seq);
            
            fprintf(1,'# Further calibration of the initial values...\n');
            model{i} = vargplvmInitDynamics(model{i},optionsDyn{i}, samd);
            
            %  to also not learn the last kernel's variance
            if numel(kern.comp) > 1 && exist('learnSecondVariance') && ~learnSecondVariance
                fprintf(1,'# The variance for %s in the dynamics is not learned!\n',kern.comp{end}.type)
                model{i}.dynamics.learnSecondVariance = 0;
                model{i}.dynamics.kern.comp{end}.inverseWidth = model{i}.dynamics.kern.comp{1}.inverseWidth/10; %%% TEMP
            end
            
            model{i}.beta=1/(0.01*var(model{i}.m(:)));
        end
        
        model{i}.dynamics.At = (model{i}.alpha)^2*model{i}.dynamics.Kt + (1-model{i}.alpha)^2*model{i}.epsilon*eye(model{i}.N);
        
        model{i}.dynamics.vardist.Sq = cell(model{i}.q,1);
        
        for q=1:model{i}.q
            LambdaH_q = model{i}.dynamics.vardist.covars(:,q).^0.5;
            Bt_q = eye(model{i}.N) + LambdaH_q*LambdaH_q'.*model{i}.dynamics.At;
            
            % Invert Bt_q
            Lbt_q = jitChol(Bt_q)';
            G1 = Lbt_q \ diag(LambdaH_q);
            G = G1*model{i}.dynamics.At;
            % Find Sq
            model{i}.dynamics.vardist.Sq{q} = model{i}.dynamics.At - G'*G;
        end
        
        % NEW!!!!!
        if var_mi < 1e-8
            warning(['Variance in model ' num2str(i) ' was too small. Setting beta to 1e+7'])
            model{i}.beta = 1e+7;
        else
            model{i}.beta = 1/((1/globalOpt.initSNR * var_mi));
        end
    end
    %model = svargplvmModelCreate(Ytr, globalOpt, options, optionsDyn);
    %modelInit = model;%%%TEMP
    
    %     %--  Unify models into a structure
    model = mvcgpdsModelCreate(model);
    model.dataSetNames = dataSetNames;
    model.experimentNo = experimentNo;
    model.dataType = dataType;
    %-- Define what level of parallelism to use (w.r.t submodels or/and w.r.t
    % datapoints).
end





if isTraining
    if parallel
        fprintf('# Parallel computations w.r.t the datapoints!\n');
        model.vardist.parallel = 1;
        for i=1:model.numModels
            model.comp{i}.vardist.parallel = 1;
        end
    end
    % Force kernel computations
    samd = [1:model.comp{1}.d];
    params = mvcgpdsExtractParam(model, samd);
    model = mvcgpdsExpandParam(model, params, samd);
    
    %%
    display = 1;
    %%%% Optimisation
    
    if addBetaPrior
        %--- Config: where and what prior to add
        meanSNR = 150;                       % Where I want the expected value of my inv gamma if it was on SNR
        priorName = 'invgamma';              % What type of prior
        scale = priorScale*model.N;    % 'strength' of prior.
        
        for i=1:length(model.comp)
            if isfield(model.comp{i}, 'mOrig'), varData = var(model.comp{i}.mOrig(:));  else  varData = var(model.comp{i}.m(:));  end
            meanB = meanSNR./varData;
            a=0.08;%1.0001; % Relatively large right-tail
            b=meanB*(a+1); % Because mode = b/(a-1)
            % Add the prior on parameter 'beta'. The prior is specified by its name
            % (priorName) and its parameters ([a,b])
            model.comp{i} = vargplvmAddParamPrior(model.comp{i}, 'beta', priorName, [a b]);
            
            % Add a scale to the prior ("strength") and save this version of the model.
            model.comp{i}.paramPriors{1}.prior.scale = scale;
        end
        params = mvcgpdsExtractParam(model, samd);
        model = mvcgpdsExpandParam(model, params, samd);
    end
    
    model.initVardist = 1; model.learnSigmaf = 0;
    model = svargplvmPropagateField(model,'initVardist', model.initVardist);
    model = svargplvmPropagateField(model,'learnSigmaf', model.learnSigmaf);
    
    model.fixEpsilon = 1;
    model.fixAlpha = 1;
    model.fixBeta = 1;
    %fix kernel parameters of \theta_t
    model.fixDynKern = 1;
    model.fixShareDynKern = 1;
    iter = 50;
    for i = 1:5
        model.onlyOptiVardist = 1;
        model.onlyOptiModel = 0;
        fprintf(1,'# Optimise the variational parameters for %d iterations...\n', iter);
        model = mvcgpdsOptimise(model, samd, display, iter); % Default: 20
        
        model.onlyOptiVardist = 0;
        model.onlyOptiModel = 1;
        fprintf(1,'# Optimise the model parameters for %d iterations...\n', iter);
        model = mvcgpdsOptimise(model, samd, display, iter); % Default: 20
        save(fileToSave,'model');
    end
    
    %optimize variational and model parameters together
    model.onlyOptiVardist = 0;
    model.onlyOptiModel = 0;
    iters = 1000;
    fprintf(1,'# Optimise variational and model parameters for %d iterations (session %d)...\n',iters,i);
    model = mvcgpdsOptimise(model, samd, display, iters);
    save(fileToSave,'model');
else
    load(fileToSave);
end
   
%%

if ~exist('resultsDynamic')  resultsDynamic = 0; end

v = 1;
modelVis = model.comp{v};

%%
if ~exist('doPredictions'), doPredictions = 1; end
if ~doPredictions
    return
end

%---------------------------- PREDICTIONS ---------------

% Set to 1 to test on the training data itself, set to 0 to use the test
% dataset.
if ~exist('testOnTraining')
    testOnTraining=0;
end

% 1 is for the HoG image features. 2 is for the pose features.
obsMod = 1; % one of the involved sub-models (possible values: 1 or 2).
infMod = setdiff(1:2, obsMod);

% Number of test points to use
numberTestPoints = 10;
if testOnTraining
    perm = randperm(model.N);
    testInd = perm(1:numberTestPoints);
else
    Yts{obsMod} = Y_test;
    Yts{infMod} = Z_test;
    perm = randperm(size(Yts{obsMod},1));
   % testInd = perm(1:numberTestPoints);
    testInd = 1:size(Yts{1},1);
    ZpredAll = zeros(size(Z_test));
    indsAll = zeros(size(Z_test,1),1);
end

scrsz = get(0,'ScreenSize');

x_star = zeros(length(testInd), size(model.X,2));

%%% Dynamics
model.dynamics.t_star = timeStampsTest;
model.comp{1}.dynamics.t_star = timeStampsTest;
model.comp{2}.dynamics.t_star = timeStampsTest;
model.comp{2}.beta = 100;

for i=1:size(Z_test,1)
    % initialize the latent points using the nearest neighbour from the training data
    dst = dist2(Y_test(i,:), Ytr{1});
    [mind, mini(i)] = min(dst);
end

vardistx = vardistCreate(model.dynamics.vardist.means(mini,:), model.q, 'gaussian');
vardistx.covars = 0.2*ones(size(vardistx.covars));
model.comp{obsMod}.vardistx = vardistx;

if isOptimizingInPrediction == 1
    % Do also reconstruction in test data
    samd = [1:model.comp{obsMod}.d];
    model.fixll = 1;
    [x, varx] = mvcgpdsOptimiseSeqDyn(model.comp{obsMod}, model, vardistx, Y_test, 1, 8, samd);
    % keep the optimized variational parameters
    %optimise obejective to get mu1_star and lambda1_star
    mu1_star = x;
    lambda1_star = varx;
    
    % update shared variational parameters mu12_satr and Sq12_star 
    Kt12_star = kernCompute(model.dynamics.kern, timeStampsTest);
    Kt1_star = kernCompute(model.comp{1}.dynamics.kern, timeStampsTest);
    At1_star = (model.comp{1}.alpha)^2*Kt1_star + (1-model.comp{1}.alpha)^2*model.comp{1}.epsilon*eye(size(Y_test,1));
    Sq12_star = cell(model.q,1);
    mu12_star = zeros(size(Y_test,1),model.q);
    for q = 1:model.q
        Sq12_star{q} = inv((1-model.comp{1}.alpha)^2*inv(At1_star) + inv(Kt12_star));
        mu12_star(:,q) = Sq12_star{q}*((1-model.comp{1}.alpha)*inv(At1_star)*mu1_star(:,q));
    end
else
    load(fileToSave);
end

numberOfNN = 1;

fprintf('# Finding the %d NN of X_* with the training X based only on the shared space.\n', numberOfNN);
ZpredAll = zeros(size(Z_test));
ZpredCovars = zeros(size(Z_test));
indsAll = zeros(size(Z_test,1),1);
indsAllOrig = zeros(size(Z_test,1),1);

for i=1:size(Z_test,1)
    [ind2,distInd] = nn_class(model.dynamics.vardist.means, mu12_star(i,:), numberOfNN, 'euclidean');
    indsAllOrig(i) = ind2(1);
    % nearest neighbour in shared X space
    [ind, distInd] = nn_class(model.dynamics.vardist.means, mu12_star(i,:), numberOfNN, 'euclidean');
    indsAll(i) = ind(1);
    ZpredMu = zeros(size(Z_test,1), size(model.comp{infMod}.y,2));
    ZpredSigma = zeros(size(Z_test,1), size(model.comp{infMod}.y,2));
    % Find p(y_*|x_*) for every x_* found from the NN
    for k=1:numberOfNN
        %nearest neighbour in shared X space
        x_cur = model.comp{2}.dynamics.vardist.means(ind(k),:);
        %option 2
        %nearest neighbour in Y space
        %x_cur = model.comp{2}.dynamics.vardist.means(mini(i),:);
        [ZpredMu(k,:) ZpredSigma(k,:)] = cgpdsPosteriorMeanVar(model.comp{infMod}, x_cur);%, varx_star(i,:)); % varx_star needed?
    end
    ZpredAll(i,:) = ZpredMu(1,:);
    ZpredCovars(i,:) = ZpredSigma(1,:);
end
fprintf('\n\n');

% Add the bias back in
ZpredAll = ZpredAll.*repmat(model.comp{2}.scale, size(ZpredAll,1), 1);
ZpredAll = ZpredAll + repmat(model.comp{2}.bias, size(ZpredAll,1), 1);

ZpredCovars = ZpredCovars.*repmat(model.comp{2}.scale.*model.comp{2}.scale, size(ZpredAll,1), 1);

NNYspace = Ytr{2}(mini,:);
NNXspace = Ytr{2}(indsAllOrig,:);
errors.NNYspace = xyzankurError(Ytr{2}(mini,:), Z_test);
errors.NNXspace = xyzankurError(Ytr{2}(indsAllOrig,:),Z_test);
errors.mvcgpds = xyzankurError(ZpredAll, Z_test);
msll.mvcgpds = calculateMSLL(Z_test,  ZpredAll, ZpredCovars);

fprintf('# NN in the Y space Error: %d\n',errors.NNYspace);
fprintf('# NN in the X space Error: %d\n',errors.NNXspace);
fprintf('# mvcgpds Error: %d\n', errors.mvcgpds);
fprintf('# mvcgpds MSLL: %d\n', msll.mvcgpds);

save(fileToSave,'model','Y_test','ZpredAll','ZpredCovars','Z_test','Yim_test','errors','msll','mu1_star','lambda1_star','mu12_star');
                                                                                                                                                                                                                                                                                                                                 
for i =1:size(Z_test,1)
    xyzankurAnimCompareMultiple(Z_test(i,:), {ZpredAll(i,:),NNXspace(i,:), NNYspace(i,:)},-1,{'Gr. Truth', 'mvcgpds', 'NN_X','NN_Y'});
    pause(0.1)
end
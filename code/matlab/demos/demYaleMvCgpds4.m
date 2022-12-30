% DEMYALEMvcgpds4 Run the Shared Var. GP-LVM on a subset of the Yale
% faces.
% DESC Run the Shared Var. GP-LVM on a subset of the Yale faces. The code
% for creating this subset out of raw images exists in comments. Unlike
% demYaleSvargplvm1, this demo is not a wrapper, it can be used as a
% standalone demo.
%
% VARGPLVM

% Fix seeds
clear 
randn('seed', 1e5);
rand('seed', 1e5);

%{
% Create the dataset out of the images.
baseDir=[localDatasetsDirectoryLarge 'CroppedYale' filesep 'CroppedYale'];
selDirs = {'04','07','26','31'};

for d=1:length(selDirs)
    dirFrom=[baseDir filesep 'yaleB' selDirs{d}];
    a=dir(dirFrom);
    counter = 0;
    for i=1:length(a)
        if length(a(i).name)>4 & strcmp(a(i).name(end-2:end),'pgm') ...
                & ~strcmp(a(i).name(end-10:end-4),'Ambient')
            im = imread([dirFrom filesep a(i).name]);
            %imagesc(im), colormap('gray'); title(a(i).name), pause
            counter = counter+1;
            Yall{d}(counter,:)=im(:)';
        end
    end
    Yall{d} = double(Yall{d});
end
height = size(im,1);
width = size(im,2);

numberOfDatasets = length(Yall);
%}

if ~exist('itNo')         ,  itNo = [500 1500 1500 1500];              end     % Default: 2000
if ~exist('indPoints')    ,  indPoints = 120;          end     % Default: 49
if ~exist('initVardistIters'), initVardistIters = 180;      end
if ~exist('mappingKern')   ,  mappingKern = {'rbfard2', 'white'}; end

% 0.1 gives around 0.5 init.covars. 1.3 biases towards 0.
if ~exist('vardistCovarsMult'),  vardistCovarsMult=1.3;                  end
if ~exist('invWidthMultDyn'),    invWidthMultDyn = 100;                     end
if ~exist('invWidthMult'),       invWidthMult = 5;                     end
% Set to empty value {} to work with toy data
if ~exist('dataSetNames')    ,    dataSetNames = {};    end
if ~exist('invWidthMult'),       invWidthMult = 5;                     end
if ~exist('dataType'), dataType = 'default'; end
if ~exist('latentDimPerModel'), latentDimPerModel = 10; end
if ~exist('experimentNo'), experimentNo = 404; end
if ~exist('doPredictions'), doPredictions = false; end
if ~exist('parallel', 'var'), parallel = false; end
% If this is true, then the model is in "D > N" mode.
if ~exist('DgtN'), DgtN = false; end
% Create initial X by doing e.g. ppca in the concatenated model.m's or by
% doing ppca in the model.m's separately and concatenate afterwards?
if ~exist('initial_X'), initial_X = 'separately'; end % Other options: 'concatenated'
% Which indices to use for training, rest for test
if ~exist('indTr'), indTr = -1; end
if ~exist('dynUsed')      ,  dynUsed = 1;             end
if ~exist('dynamicKern')   ,     dynamicKern = {'rbf', 'white', 'bias'}; end
if ~exist('isTraining') ,  isTraining = 1;      end

enableParallelism = 0;

%{
dataType = 'Yale4Sets';
dataSetNames = 'YaleSubset4_2';
[Y,lbls]=svargplvmLoadData(dataSetNames);
N1 = size(Y{1},1);
Yall{1} = [Y{1};Y{2}];
Yall{2} = [Y{3};Y{4}];
% Randomly exchange rows between the two person faces involved in the
% second dataset (so, they still have the same lighting angle)
if shuffleData
    r = rand(N1,1);
    r = r > 0.5;
    for i = find(r)
        temp = Yall{2}(i,:);
        Yall{2}(i,:) = Yall{2}(i+N1,:);
        Yall{2}(i+N1,:) = temp;
    end
end
clear Y;
%}


% Shuffle one of the two datasets but maintain the correct correspondance,
% i.e. the corresopnding (y_i,z_i) pairs should still be from the same
% angle.
dataType = 'Yale6Sets';
dataSetNames = 'YaleSubset6_1';
try
    [Y,lbls]=svargplvmLoadData(dataSetNames);
catch
    load YaleSubset6;
    lbls = [height width];    
end

if exist('pyramid','var') % extract pyramid representation of the images
    if pyramid
        for e=1:size(Y,2)
            Y{e} = im2pyramid(Y{e}, lbls(1), lbls(2), 4);
        end
    end
end

N1 = size(Y{1},1);
Yall{2} = [Y{4};Y{5};Y{6}];
identities{2}=[ones(N1,1) 2*ones(N1,1) 3*ones(N1,1)];

numSubsets = 3;
Yall{1} = zeros(numSubsets*N1, size(Y{1},2));
for i=1:N1
    perm = randperm(numSubsets);
    counter = 0;
    for j=perm
        Yall{1}(i+counter*N1,:) = Y{j}(i,:);
        identities{1}(i+counter*N1) = j;
        counter = counter+1;
    end
end


clear Y;


numberOfDatasets = length(Yall);
height = lbls(1); width = lbls(2);


%{
for i=1:size(Yall{1},1)
    for d=1:numberOfDatasets
        subplot(1,numberOfDatasets,d)
        imagesc(reshape(Yall{d}(i,:),height, width)), title(num2str(identities{d}(i))),colormap('gray');
        axis off
    end
    if i==1
        pause
    else
        pause(0.5)
    end
end
%}




%-- Load datasets
for i=1:numberOfDatasets
    Y = Yall{i};
    dims{i} = size(Y,2);
    N{i} = size(Y,1);
    if indTr == -1
        indTr = 1:N{i};
    end
    indTs = setdiff(1:size(Y,1), indTr);
    Ytr{i} = Y(indTr,:); 
    Yts{i} = Y(indTs,:);
    t{i} = linspace(0, 2*pi, size(Y, 1)+1)'; t{i} = t{i}(1:end-1, 1);
    timeStampsTraining{i} = t{i}(indTr,1); %timeStampsTest = t(indTs,1);
    d{i} = size(Ytr{i}, 2);
end

for i=2:numberOfDatasets
    if N{i} ~= N{i-1}
        error('The number of observations in each dataset must be the same!');
    end
end

%%

%-- Options for the models
for i=1:numberOfDatasets
    % Set up models
    options{i} = vargplvmOptions('dtcvar');
    options{i}.kern = mappingKern; %{'rbfard2', 'bias', 'white'};
    %indPoints = 80; %%%%%
    options{i}.numActive = indPoints;
    options{i}.optimiser = 'scg2';
    if ~DgtN
        options{i}.enableDgtN = false;
    end
    % !!!!! Be careful to use the same type of scaling and bias for all
    % models!!!
    
    % scale = std(Ytr);
    % scale(find(scale==0)) = 1;
    %options.scaleVal = mean(std(Ytr));
    options{i}.scaleVal = sqrt(var(Ytr{i}(:)));
end

%-------------- INIT LATENT SPACE ---%
for i=1:length(Ytr)
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
    
    %    mAll = [mAll m{i}]; % Concatenation (doesn't work if different sizes)
end

X_init_temp = cell(2,1);
if strcmp(initial_X, 'separately')
    fprintf('# Initialising X by performing ppca in each observed (scaled) dataset separately and then concatenating...\n');
    X_init_temp{1} = ppcaEmbed(m{1},latentDimPerModel);
    X_init_temp{2} = ppcaEmbed(m{2},latentDimPerModel);
    X_init_share = [X_init_temp{1} X_init_temp{2}];
else
    fprintf('# Initialising X by performing ppca in concatenated observed (scaled) data...\n');
    X_init_share = ppcaEmbed([m{1} m{2}], latentDimPerModel*2);
end
%-----------------

latentDim = size(X_init_share,2);

% Free up some memory
clear('Y')



%-- Create the sub-models: Assume that for each dataset we have one model.
% This can be changed later, as long as we find a reasonable way to
% initialise the latent spaces.
for i=1:numberOfDatasets
    %---- Here put some code to assign X to the global common X which must
    % be created by doing pca in the concatenation of Y's...After this
    % point, model{i}.X will be the same for all i's. TODO...
    fprintf(1,'# Creating the model...\n');
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
    %inpScales(:) = max(inpScales); % Optional!!!!!
    model{i}.kern_v.comp{1}.inputScales = inpScales;
    model{i}.kern_u.comp{1}.inputScales = inpScales;
    
    if strcmp(model{i}.kern_v.type, 'rbfardjit')
        model{i}.kern_v.inputScales = model{i}.kern_v.comp{1}.inputScales;
    end
    if strcmp(model{i}.kern_u.type, 'rbfardjit')
        model{i}.kern_u.inputScales = model{i}.kern_u.comp{1}.inputScales;
    end
    
    
    params = cgpdsExtractParam(model{i}, samd);
    model{i} = cgpdsExpandParam(model{i}, params, samd);
    model{i}.vardist.covars = 0.5*ones(size(model{i}.vardist.covars)) + 0.001*randn(size(model{i}.vardist.covars));
    
    %-------- Add dynamics to the model -----
    if  dynUsed
        fprintf(1,'# Adding dynamics to the model...\n');
        optionsDyn{i}.type = 'vargpTime';
        optionsDyn{i}.t =timeStampsTraining;
        optionsDyn{i}.inverseWidth=invWidthMultDyn; % Default: 100
        optionsDyn{i}.initX = X_init_share;
        kern = kernCreate(optionsDyn{i}.t{i}, dynamicKern); % Default: {'rbf','white','bias'}
        
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
            kern.comp{1}.inverseWidth = optionsDyn{i}.inverseWidth./(((max(t{i})-min(t{i}))).^2);
            kern.comp{1}.variance = 1;
        end
        
        optionsDyn{i}.kern = kern;
        optionsDyn{i}.vardistCovars = vardistCovarsMult; % 0.23 gives true vardist.covars around 0.5 (DEFAULT: 0.23) for the ocean dataset
        
        % Fill in with default values whatever is not already set
        optionsDyn{i} = vargplvmOptionsDyn(optionsDyn{i});
        
        model{i} = vargplvmAddDynamics(model{i}, 'vargpTime', samd, optionsDyn{i}, optionsDyn{i}.t{i}, 0, 0,optionsDyn{i}.seq);
        
        fprintf(1,'# Further calibration of the initial values...\n');
        model{i} = vargplvmInitDynamics(model{i},optionsDyn{i}, samd);
        
        %  to also not learn the last kernel's variance
        if numel(kern.comp) > 1 && exist('learnSecondVariance') && ~learnSecondVariance
            fprintf(1,'# The variance for %s in the dynamics is not learned!\n',kern.comp{end}.type)
            model{i}.dynamics.learnSecondVariance = 0;
            model{i}.dynamics.kern.comp{end}.inverseWidth = model{i}.dynamics.kern.comp{1}.inverseWidth/10; %%% TEMP
        end
        
        model{i}.beta=1/(0.01*var(model{i}.m(:)));
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
    end
    
    model{i}.beta = 1e+7;
    %     prunedModelInit{i} = vargplvmPruneModel(model{i});
    %disp(model{i}.vardist.covars)
end




%modelInit = model;%%%TEMP

%--  Unify models into a structure
svargplvm_init
model = svargplvmModelCreate(model);

model.comp{1}.W = 1e-6*model.comp{1}.W;
model.comp{2}.W = 1e-6*model.comp{2}.W;

model.dataSetNames = dataSetNames;
model.experimentNo = experimentNo;
model.dataType = dataType;
%%---
capName = dataType;
capName(1) = upper(capName(1));
modelType = model.type;
modelType(1) = upper(modelType(1));
fileToSave = ['dem' capName modelType num2str(experimentNo) '.mat'];
save(fileToSave,'model');
%%---


%-- Define what level of parallelism to use (w.r.t submodels or/and w.r.t
% datapoints).
%{
fprintf('# Parallel computations w.r.t the submodels!\n');
model.parallel = 1;
model = svargplvmPropagateField(model,'parallel', 1);
%
fprintf('# Parallel computations w.r.t the datapoints!\n');
model.vardist.parallel = 1;
for i=1:model.numModels
    model.comp{i}.vardist.parallel = 1;
end
%}


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
    % do not learn beta and sigma_f for few iterations for intitialization
    
    model.initVardist = 1; model.learnSigmaf = 0;
    model = svargplvmPropagateField(model,'initVardist', model.initVardist);
    model = svargplvmPropagateField(model,'learnSigmaf', model.learnSigmaf);
    
    model.fixW = 1;
    model.fixEpsilon = 1;
    model.fixAlpha = 1;
    model.fixBeta = 1;
    model.fixDynKern = 1;
    model.fixShareDynKern = 1;
    model.fixU = 1;
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
    
    %optimize all parameters together
    model.onlyOptiVardist = 0;
    model.onlyOptiModel = 0;
    iters = 1000;
    fprintf(1,'# Optimise all parameters for %d iterations (session %d)...\n',iters,i);
    model = mvcgpdsOptimise(model, samd, display, iters);
    save(fileToSave,'model');
else
    load(fileToSave);
end

%%

%---------------------------- PREDICTIONS ---------------
if ~doPredictions
    return
end


obsMod = 1; % one of the involved sub-models (the one for which we have the data)
infMod = setdiff(1:2, obsMod);

if ~exist('testOnTraining')
    testOnTraining=1;
end

numberTestPoints = 10;
if testOnTraining
    perm = randperm(model.N);
    testInd = perm(1:numberTestPoints);
else
    perm = randperm(size(Yts{obsMod},1));
    testInd = perm(1:numberTestPoints);
end

scrsz = get(0,'ScreenSize');

for i=1:length(testInd)
    curInd = testInd(i);
    fprintf('# Testing indice number %d ', curInd);
    if testOnTraining
        fprintf('taken from the training set\n');
        y_star = model.comp{obsMod}.y(curInd,:);
        x_star = model.comp{obsMod}.vardist.means(curInd,:);
        varx_star = model.comp{obsMod}.vardist.covars(curInd,:);
    else
        fprintf('taken from the test set\n');
        y_star = Yts{obsMod}(curInd,:);
        z_star = Yts{infMod}(curInd,:);
        dst = dist2(y_star, model.comp{obsMod}.y);
        [mind, mini] = min(dst);
        
        Init(i,:) = model.vardist.means(mini,:);
        vardistx = vardistCreate(model.comp{obsMod}.vardist.means(mini,:), model.q, 'gaussian');
        vardistx.covars = model.comp{obsMod}.vardist.covars(mini,:);
        model.comp{obsMod}.vardistx = vardistx;
        display=1;
        iters = 250;
        % Find p(X_* | Y_*) which is approximated by q(X_*)
        [x_star, varx_star, modelUpdated] = vargplvmOptimisePoint(model.comp{obsMod}, vardistx, y_star, display, iters);%%%
%         samd = [1:model.comp{obsMod}.d];
%         model.fixlikelihood = 1;
%         model.fixcovars = 1;
%         [x_star, varx_star] = mvcgpdsOptimiseSeqDyn(model.comp{obsMod}, model, vardistx, y_star, 1, iters, samd);
%         
    end
    
    mu1_star = x_star;
    lambda1_star = varx_star;
     
    Kt12_star = kernCompute(model.dynamics.kern, timeStampsTest);
    Kt1_star = kernCompute(model.comp{1}.dynamics.kern, timeStampsTest);
    At1_star = (model.comp{1}.alpha)^2*Kt1_star + (1-model.comp{1}.alpha)^2*model.comp{1}.epsilon*eye(size(Y_test,1));
    Sq12_star = cell(model.q,1);
    mu12_star = zeros(size(Y_test,1),model.q);
    for q = 1:model.q
        Sq12_star{q} = inv((1-model.comp{1}.alpha)^2*inv(At1_star) + inv(Kt12_star));
        mu12_star(:,q) = Sq12_star{q}*((1-model.comp{1}.alpha)*inv(At1_star)*mu1_star(:,q));
    end
    
    numberOfNN = 9;
    % Now we selected a datapoint X_* by taking into account only the
    % private dimensions for Y. Now, based on the shared dimensions of
    % that, we select the closest (in a NN manner) X from the training data.
    fprintf('# Finding the %d NN of X_* with the training X based only on the shared dims.\n', numberOfNN);
    [ind, distInd] = nn_class(model.dynamics.vardist.means, mu12_star(i,:), numberOfNN, 'euclidean');
        
    ZpredMu = zeros(length(ind), size(model.comp{infMod}.y,2));
    ZpredSigma = zeros(length(ind), size(model.comp{infMod}.y,2));
      
    
    % Find p(y_*|x_*) for every x_* found from the NN
    fprintf('# Predicting images from the NN of X_* ');
    for k=1:numberOfNN
        fprintf('.');
        x_cur = model.comp{2}.dynamics.vardist.means(ind(k),:);
            %option 2
%             x_cur = model.comp{2}.dynamics.vardist.means(mini(i),:)
        %x_cur(sharedDims) = x_star(sharedDims); %%% OPTIONAL!!!
        %[ZpredMu(k,:), ZpredSigma(k,:)] = vargplvmPosteriorMeanVar(model.comp{infMod}, model.X(ind(k),:));
        ZpredMu(k,:) = vargplvmPosteriorMeanVar(model.comp{infMod}, x_cur);
    end
    fprintf('\n\n');
    ZpredMu = ZpredMu.*repmat(model.comp{2}.scale, size(ZpredMu,1), 1);
    ZpredMu = ZpredMu + repmat(model.comp{2}.bias, size(ZpredMu,1), 1);
    
    
    %-- Plots
    % Open a big figure (first 2 args control the position, last 2 control
    % the size)
    figure('Position',[scrsz(3)/100.86 scrsz(4)/6.666 scrsz(3)/1.0457 scrsz(4)/1.0682],...
        'Name',['Fig: ' num2str(i) ' (Exp: ' num2str(experimentNo) ')'],'NumberTitle','off')
    numRows = 2;
    
    if testOnTraining
        numCols = ceil((numberOfNN+1)/numRows);
        plotCounter = 1;
    else
        % For the real test image!
        numCols = ceil((numberOfNN+2)/numRows);
        plotCounter = 2;
    end
    subplot(numRows, numCols, 1)
    imagesc(reshape(y_star,height,width)), title(['Original y (image #' num2str(curInd) ')']), colormap('gray')
    
    if ~testOnTraining
         subplot(numRows, numCols, 2)
         imagesc(reshape(z_star,height,width)), title(['Corresponding z (image #' num2str(curInd) ')']), colormap('gray')
    end
    
    for k=1:numberOfNN
        subplot(numRows, numCols, k+plotCounter)
        imagesc(reshape(ZpredMu(k,:), height, width)), title(['NN #' num2str(k)]), colormap('gray');
   end
        
        
    %{
    indYnn = [];
    % Do a NN on the DATA space, for every predicted output.
    for j=1:size(ZpredMu,1)
        [indYnn(j), distInd] = nn_class(model.comp{infMod}.y, ZpredMu(j,:),1,'euclidean');
    end
    for j=1:length(indYnn)
        figure, imagesc(reshape(model.comp{infMod}.y(indYnn(j),:),height, width)), title([num2str(j)]), colormap('gray')
    end
    %}
end

if ~testOnTraining
    errsumFull = sum((ZpredMu - Yts).^2);
    errorFull = mean(errsumFull);
end


%{
% OLD CODE!!
if testOnTraining
  %  i = size(model.comp{obsMod}.y,1); % last observation
   i=23;
   %i=120;
    % Find p(X_* | Y_*): If Y* is not taken from the tr. data, then this step
    % must be an optimisation step of q(x*). X_* and Y_* here refer to the
    % spaces of the submodel obsMod.
    y_star = model.comp{obsMod}.y(i);
    x_star = model.comp{obsMod}.vardist.means(i,:);
    varx_star = model.comp{obsMod}.vardist.covars(i,:);
    % Find the 10 closest latent points to the x_*, based only on the
    % sharedDimensions (since we test on a training point, one of the
    % distances is going to be 0, i.e. the test point itself).
    [ind, distInd] = nn_class(model.X(:,sharedDims), x_star(:,sharedDims), 5, 'euclidean');
    
    ZpredMu = zeros(length(ind), size(model.comp{infMod}.y,2));
    ZpredSigma = zeros(length(ind), size(model.comp{infMod}.y,2));
    
    % Find p(y_*|x_*) for every x_* found from the NN
    for k=1:length(ind)
        [ZpredMu(k,:), ZpredSigma(k,:)] = vargplvmPosteriorMeanVar(model.comp{infMod}, model.X(ind(k),:));
    end
    
    indYnn = [];
    % Do a NN on the DATA space, for every predicted output.
    for j=1:size(ZpredMu,1)
        [indYnn(j), distInd] = nn_class(model.comp{infMod}.y, ZpredMu(j,:),1,'euclidean');
    end
    for j=1:length(indYnn)
        figure, imagesc(reshape(model.comp{infMod}.y(indYnn(j),:),height, width)), title([num2str(j)]), colormap('gray')
    end
    figure, imagesc(reshape(model.comp{obsMod}.y(i,:),height,width)), title('Original'), colormap('gray')
else
    testInd = 25;
    Yts = Y_test(testInd,:); % Now this is a matrix
    for i=1:size(Yts,1)
        % initialize the latent points using the nearest neighbour
        % from the training data
        dst = dist2(Yts(i,:), model.comp{obsMod}.y);
        [mind, mini] = min(dst);
        
        Init(i,:) = model.vardist.means(mini,:);
        vardistx = vardistCreate(model.comp{obsMod}.vardist.means(mini,:), model.q, 'gaussian');
        vardistx.covars = model.comp{obsMod}.vardist.covars(mini,:);
        model.comp{obsMod}.vardistx = vardistx;
        display=1;
        iters = 100;
        % Find p(X_* | Y_*) which is approximated by q(X_*)
        [x_star, varx_star, modelUpdated] = vargplvmOptimisePoint(model.comp{obsMod}, vardistx, Yts(i,:), display, iters);%%%
        numberOfNN = 10;
        % Now we selected a datapoint X_* by taking into account only the
        % private dimensions for Y. Now, based on the shared dimensions of
        % that, we select the closest (in a NN manner) X from the training data.
        [ind, distInd] = nn_class(model.X(:,sharedDims), x_star(:,sharedDims), numberOfNN, 'euclidean');
        
        x = model.X(ind,:);
        mu = vargplvmPosteriorMeanVar(model.comp{infMod}, x);
        
        ZpredK{i} = mu;
        
        ZpredMu(i,:) = mu(1);
        
        %ZpredSigma(i,:) = sigma;
    end
    
    errsumFull = sum((ZpredMu - Z_test(testInd,:)).^2);
    errorFull = mean(errsumFull);
end

%}
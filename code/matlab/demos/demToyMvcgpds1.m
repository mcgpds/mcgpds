% DEMOMvcgpds1 A simple demo of mvcgpds.
% DESC This is a simple demo which uses a toy dataset that considers two
% modalities: each of the modalities has one private 1-dimensional signal and they
% both share a 1-dimensional shared signal. The final dataset is mapped to 15-dimensional.
% The demo allows training a MvCGPDS model so that recover the true signals.

% Fix seeds
randn('seed', 1e5);
rand('seed', 1e5);

%% ---  High level options for the demo
%-- Mapping kernel: One cell for each modality. Compound kernels (cells
%themselves) allowed, e.g. {k1, k2} means k1+k2.
if ~exist('baseKern', 'var'), baseKern = {{'linard2', 'white', 'bias'}, {'linard2', 'white', 'bias'}}; end
%if ~exist('baseKern', 'var'), baseKern = {'linard2','linard2'}; end
% if ~exist('baseKern', 'var'), baseKern = {{'rbfard2','white', 'bias'},{'rbfard2','white', 'bias'}}; end
% if ~exist('baseKern', 'var'), baseKern = {{'rbfard2','linard2'},{'rbfard2','linard2'}}; end
%if ~exist('baseKern', 'var'), b30aseKern = {'rbfardjit','rbfardjit'}; end
%-- Initialisation of latent space:
if ~exist('latentDimPerModel', 'var'), latentDimPerModel = 1; end % Only used if initial_X is 'separately'
if ~exist('latentDim', 'var'), latentDim = 1; end % Only used if initial_X is 'concatenated'
if ~exist('initial_X', 'var'), initial_X = 'separately'; end
if ~exist('printPlot', 'var'), printPlot = false; end
if ~exist('experimentNo', 'var'), experimentNo = 1003; end
if ~exist('isTraining', 'var'), isTraining = 1; end

%% Create toy dataset
% generates a row vector y of 100 points linearly spaced between and including a and b
% alpha = linspace(-1, 1, num_points);
% Scale and center data, normalize the data
%% -----------------expriment notes-------------------------------------
% Z3 = scaleData(cos(alpha)'.^2, 2);
% experimentNo = 1001
%--------------------------------------------------------------------
% Z3 = scaleData(2*cos(2*alpha)' + 2*sin(2*alpha)', 2);
% experimentNo = 1002
%--------------------------------------------------------------------
% Z1 = scaleData(cos(pi*pi*alpha)', 2);
% Z2 = scaleData(cos(sqrt(5)*pi*alpha)', 2);
% Z3 = scaleData(sin(2*pi*alpha)', 2);
% alpha = linspace(-1,1,100);
% experimentNo = 1003
%--------------------------------------------------------------------
%%
alpha = linspace(-1,1,100);%30,43,77,217,237,80
Z1 = scaleData(cos(pi * pi * alpha)', 2);
Z2 = scaleData(cos(sqrt(5) * pi * alpha)', 2);
Z3 = scaleData(sin(2 * pi * alpha)', 2);
%%
% % alpha = linspace(0,4*pi,100);%30,43,77,217,237,80
% alpha = linspace(-1,1,100);%30,43,77,217,237,80
% % Scale and center data, normalize the data
% Z1 = scaleData(cos(alpha)', 2);
% Z2 = scaleData(sin(alpha)', 2);
% Z3 = scaleData(cos(alpha)'+sin(alpha)', 2); 
%%
noiseLevel = 0.1; % Default: 0.1
% Map 1-dim to 10-dim and add some noise, Here N = 100, Z2 is a N*1 matrix, and
% Z2p is a N*10 matrix.
Z2p = Z2 * rand(1, 10);
Z2p = Z2p + noiseLevel .* randn(size(Z2p));
Z1p = Z1 * rand(1, 10);
Z1p = Z1p + noiseLevel .* randn(size(Z1p));
Z3p = Z3 * rand(1, 5); %
Z3p = Z3p + noiseLevel .* randn(size(Z3p)); %

Yall{1} = [Z1p Z3p];
Yall{2} = [Z2p Z3p];
dataSetNames = {'fols_cos', 'fols_sin'};
M = 2; % number of modalities

% This script initialises the options structure 'globalOpt'.
svargplvm_init;

%% Split data into training and test sets.
for i = 1:M
    Y = Yall{i};
    dims{i} = size(Y, 2);
    N{i} = size(Y, 1);
    indTr = globalOpt.indTr;
    if indTr == -1, indTr = 1:N{i}; end

    if ~exist('Yts', 'var')
        indTs = setdiff(1:size(Y, 1), indTr);
        Yts{i} = Y(indTs, :);
    end

    Ytr{i} = Y(indTr, :);
end

t = linspace(-1, 1, size(Y, 1) + 1)'; t = t(1:end - 1, 1);
timeStampsTraining = t(indTr, 1); %timeStampsTest = t(indTs,1);
clear('Y')

for i = 2:M

    if N{i} ~= N{i - 1}
        error('The number of observations in each dataset must be the same!');
    end

end

%% -- Create model
options = mvcgpdsOptions(Ytr, globalOpt);

% Allow for constraining the latent space with a prior which couples it.
% This can e.g be a termporal one (type: 'vargpTime') for which timestamps can
% be given
if ~isempty(globalOpt.dynamicsConstrainType)
    optionsDyn.type = 'vargpTime';
    optionsDyn.inverseWidth = 30;
    optionsDyn.initX = globalOpt.initX;
    optionsDyn.constrainType = globalOpt.dynamicsConstrainType;

    if exist('timeStampsTraining', 'var')
        optionsDyn.t = timeStampsTraining;
    end

    if exist('labelsTrain', 'var') && ~isempty(labelsTrain)
        optionsDyn.labels = labelsTrain;
    end

else
    optionsDyn = [];
end

model = mvcgpdsModelCreate(Ytr, globalOpt, options, optionsDyn);

if ~isfield(globalOpt, 'saveName') || isempty(globalOpt.saveName)
    modelType = model.type;
    modelType(1) = upper(modelType(1));
    globalOpt.saveName = ['dem' modelType num2str(globalOpt.experimentNo) '.mat'];
end

model.saveName = globalOpt.saveName;
model.globalOpt = globalOpt;
model.options = options;

% Force kernel computations
% samd is used for stochastic optimization, but we do not use
% stochasctic optimization here, so samd is the collection of all
% dimensions.
samd = [1:model.comp{1}.d];
%extract model parameters
params = mvcgpdsExtractParam(model, samd);
%update model using the extracted parameters
model = mvcgpdsExpandParam(model, params, samd);
filename = [pwd filesep 'code' filesep 'matlab' filesep 'mvcgpdsResultMat' filesep 'demToyMvcgpds' num2str(experimentNo) '.mat'];

if isTraining == 1
    display = 1;
    model.fixBeta = 0; % fix the parameter beta
    % fix the parameter of K_tt
    model.fixDynKern = 0; %kernel parameters of private K_tt
    model.fixShareDynKern = 1; % kernel parameters of shared K_tt

    %optimise the model
    model.fixAlpha = 1; % fix the parameter alpha
    iters = 50;
    fprintf(1, '# Optimising parameters for %d iterations...\n', iters);
    model = mvcgpdsOptimise(model, samd, display, iters);
    save(filename, 'model');

    model.fixBeta = 0; % fix the parameter beta
    % fix the parameter of K_tt
    model.fixDynKern = 0; %kernel parameters of private K_tt
    model.fixShareDynKern = 1; % kernel parameters of shared K_tt
    model.fixAlpha = 0; %optimize the parameter alpha with other parameters
    iters = 30;
    fprintf(1, '# Optimising parameters for %d iterations...\n', iters);
    model = mvcgpdsOptimise(model, samd, display, iters);

    model.fixBeta = 1; % fix the parameter beta
    % fix the parameter of K_tt
    model.fixDynKern = 1; %kernel parameters of private K_tt
    model.fixShareDynKern = 0; % kernel parameters of shared K_tt
    model.fixAlpha = 1; %optimize the parameter alpha with other parameters
    iters = 30;
    fprintf(1, '# Optimising parameters for %d iterations...\n', iters);
    model = mvcgpdsOptimise(model, samd, display, iters);
    save(filename, 'model');
else
    load(filename);
end

%% Results
figure
plot(alpha, Z3, 'r', 'LineWidth', 1.5);
hold on
plot(alpha, model.dynamics.vardist.means / max(model.dynamics.vardist.means), 'b', 'LineWidth', 1.5)
[nms, nidx] = newmeans(model.dynamics,alpha);
plot(nidx, nms / max(nms), 'g', 'LineWidth', 1.5)
legend('true shared signal', 'recoverd shared signal');
ylim([-1, 1]);
saveas(gcf, [pwd filesep 'code' filesep 'shareSignal' num2str(experimentNo) '.eps'], 'psc2');
saveas(gcf, [pwd filesep 'code' filesep 'shareSignal' num2str(experimentNo) '.jpg']);

figure
plot(alpha, Z1, 'r', 'LineWidth', 1.5);
hold on
plot(alpha, model.comp{1}.dynamics.vardist.means / max(model.comp{1}.dynamics.vardist.means), 'b', 'LineWidth', 1.5)
[nms, nidx] = newmeans(model.comp{1}.dynamics,alpha);
% plot(nidx, nms / max(nms), 'g', 'LineWidth', 1.5)
plot(nidx, nms, 'g', 'LineWidth', 1.5)
legend('true private signal in view 1', 'recoverd private signal in view 1')
ylim([-1, 1]);
saveas(gcf, [pwd filesep 'code' filesep 'privateSignal1' num2str(experimentNo) '.eps'], 'psc2');
saveas(gcf, [pwd filesep 'code' filesep 'privateSignal1' num2str(experimentNo) '.jpg']);

figure
plot(alpha, Z2, 'r', 'LineWidth', 1.5);
hold on
plot(alpha, model.comp{2}.dynamics.vardist.means / max(model.comp{2}.dynamics.vardist.means), 'b', 'LineWidth', 1.5)
[nms, nidx] = newmeans(model.comp{2}.dynamics,alpha);
plot(nidx, nms / max(nms), 'g', 'LineWidth', 1.5)
legend('true private signal in view 2', 'recoverd private signal in view 2')
ylim([-1, 1])
saveas(gcf, [pwd filesep 'code' filesep 'privateSignal2' num2str(experimentNo) '.eps'], 'psc2');
saveas(gcf, [pwd filesep 'code' filesep 'privateSignal2' num2str(experimentNo) '.jpg']);

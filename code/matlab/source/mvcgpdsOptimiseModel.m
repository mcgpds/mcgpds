function model = mvcgpdsOptimiseModel(model, varargin)

% mvcgpdsOptimiseModel Optimise the mvcgpds wrapper with options

% 
% COPYRIGHT: Andreas C. Damianou, 2012, 2013



pruneModel = true; 
saveModel = true;

if isfield(model, 'saveName')
    if strcmp(model.saveName, 'noSave')
        saveModel = false;
    end
end

globalOpt = model.globalOpt;


if nargin > 2
    pruneModel = varargin{1};
    if length(varargin) > 1
        saveModel = varargin{2};
    end

    if length(varargin) > 2
        globalOpt.initVardistIters = varargin{3}{1};
        globalOpt.itNo = varargin{3}{2};
    end
end


if ~isfield(model, 'iters')
    model.iters=0;
end
if ~isfield(model, 'initVardistIters')
    model.initVardistIters = 0;
end
display = 1;


i=1;
samd = [1:model.comp{1}.d];
while ~isempty(globalOpt.initVardistIters(i:end)) || ~isempty(globalOpt.itNo(i:end))
    % do not learn beta for few iterations for intitilization
    if  ~isempty(globalOpt.initVardistIters(i:end)) && globalOpt.initVardistIters(i)
        model.initVardist = 1; model.learnSigmaf = 0;
        model = svargplvmPropagateField(model,'initVardist', model.initVardist);
        model = svargplvmPropagateField(model,'learnSigmaf', model.learnSigmaf);
        model.onlyOptiVardist = 1;
        fprintf(1,'# Intitiliazing the variational distribution for %d iterations...\n', globalOpt.initVardistIters);
        model = mvcgpdsOptimise(model, samd, display, globalOpt.initVardistIters); % Default: 20
        model.initVardistIters = model.initVardistIters + globalOpt.initVardistIters(i);
        if saveModel
            fprintf('# Saving model after optimising the var. distr. for %d iterations...\n\n', globalOpt.initVardistIters(i))
            if pruneModel
                modelPruned = svargplvmPruneModel(model);
                vargplvmWriteResult(modelPruned, modelPruned.type, globalOpt.dataSetName, globalOpt.experimentNo);
            else
                vargplvmWriteResult(model, model.type, globalOpt.dataSetName, globalOpt.experimentNo);
            end
        end
    end
    
    % Optimise the model.
    model.learnBeta = 1;
    model.initVardist = 0;
    model.learnSigmaf = 1;

    model.date = date;
    if  ~isempty(globalOpt.itNo(i:end)) && globalOpt.itNo(i)
        model = svargplvmPropagateField(model,'initVardist', model.initVardist);
        model = svargplvmPropagateField(model,'learnBeta', model.learnBeta);
        model = svargplvmPropagateField(model,'learnSigmaf', model.learnSigmaf);

        iters = globalOpt.itNo(i); % Default: 1000
        model.onlyOptiVardist = 0;
        model.onlyOptiModel = 1;
        fprintf(1,'# Optimising the model for %d iterations (session %d)...\n',iters,i);
        model = mvcgpdsOptimise(model, samd, display, iters);
        for j=1:length(model.comp)
            fprintf('# SNR%d = %.6f\n', j, vargplvmShowSNR(model.comp{j},0))
        end
        model.iters = model.iters + iters;
        % Save the results.
        if saveModel
            fprintf(1,'# Saving model after optimising for %d iterations\n\n',iters)
            if pruneModel
                modelPruned = svargplvmPruneModel(model);
                vargplvmWriteResult(modelPruned, modelPruned.type, globalOpt.dataSetName, globalOpt.experimentNo);
            else
                vargplvmWriteResult(model, model.type, globalOpt.dataSetName, globalOpt.experimentNo);
            end
        end
    end
    i = i+1;
    
    %optimize all parameters together
    model.onlyOptiVardist = 0;
    model.onlyOptiModel = 0;
    iters = 1000;
    fprintf(1,'# Optimising all parameters for %d iterations (session %d)...\n',iters,i);
    model = mvcgpdsOptimise(model, samd, display, iters);
    
end





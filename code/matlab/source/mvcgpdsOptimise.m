function [model, grChek] = mvcgpdsOptimise(model, samd, display, iters, varargin)

% mvcgpdsOptimiseModel Optimise the mvcgpds.
% FORMAT
% DESC takes a given multi-view cgpds model structure and optimises with
% respect to parameters and latent positions.
% ARG model : the model to be optimised.
% ARG display : flag dictating whether or not to display
% optimisation progress (set to greater than zero) (default value 1).
% ARG iters : number of iterations to run the optimiser
% for (default value 2000).
% RETURN model : the optimised model.
%
% SEEALSO : mvcgpdsModelCreate, mvcgpdsLogLikelihood,
% mvcgpdsLogLikeGradients, mvcgpdsObjective, mvcgpdsGradient,
% mvcgpdsOptimiseModel
%


grChek = [];

if nargin < 3
    iters = 2000;
    if nargin < 2
        display = 1;
    end
end

options = optOptions;
params = modelExtractParam(model);

if isfield(model, 'throwSNRError') && model.throwSNRError
    throwSNRError = true;
else
    throwSNRError = false;
end

%%%%
if length(varargin) == 2
    if strcmp(varargin{1}, 'gradcheck')
        assert(islogical(varargin{2}));
        %options(9) = varargin{2};
        doGradchek = varargin{2};
        if doGradchek
            [gradient, delta] = feval('gradchek', params, @modelObjective, @modelGradient, model);
            deltaf = gradient - delta;
            d=norm(deltaf - gradient)/norm(gradient + deltaf); %%
            d1=norm(deltaf - gradient,1)/norm(gradient + deltaf,1); %%
            grRatio = sum(abs(gradient ./ deltaf)) / length(deltaf);
            fprintf(1,' Norm1 difference: %d\n Norm2 difference: %d\n Ratio: %d\n',d1,d, grRatio);
            grChek = {delta, gradient, deltaf, d, d1};
        else
            grChek = [];
        end
    end
end


options(2) = 0.1*options(2);
options(3) = 0.1*options(3);

if display
    options(1) = 1;
end
options(14) = iters;

if iters > 0
    if isfield(model, 'optimiser')
        optim = str2func(model.optimiser);
    else
        optim = str2func('scg');
    end
    % NETLAB style optimization.
    % use scg to optimize the parameters, the mvcgpdsObjective fuction computes the
    % evidence lower bound, and the mvcgpdsGradient fuction computes
    % gradient w.r.t parameters.
    params = optim('mvcgpdsObjective', params,  options,  'mvcgpdsGradient', model, samd);
    % update the model after optimizing parameters
    model = mvcgpdsExpandParam(model, params, samd);
%     svargplvmCheckSNR(svargplvmSNR(model), [], [], throwSNRError);
end
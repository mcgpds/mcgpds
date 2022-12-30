function [X varX] = mvcgpdsOptimiseSeqDyn(model, modelAll, vardistx, y, display, iters, samd)

if nargin < 5
  iters = 2000;
  %if nargin < 5
    display = true;
  %end
end

options = optOptions;
if display
  options(1) = 1;
  %options(9) = 1;
end
options(14) = iters;


if isfield(model, 'optimiser')
  optim = str2func(model.optimiser);
else
  optim = str2func('scg');
end

x = modelExtractParam(vardistx);

x = optim('mvcgpdsSeqDynObjective', x,  options, ...
            'mvcgpdsSeqDynGradient', model, modelAll, y, samd);

vardistx = vardistExpandParam(vardistx,x);
X = vardistx.means;
varX = vardistx.covars;

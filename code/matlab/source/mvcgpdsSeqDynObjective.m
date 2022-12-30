function f = mvcgpdsSeqDynObjective(x, model, modelAll, y, samd)
% mvcgpdsSeqDynObjective Log-likelihood of the mvcgpds for test data.

vardistx = model.vardistx;
vardistx = vardistExpandParam(vardistx, x);

f = - mvcgpdsSeqDynLogLikelihood(model, modelAll, vardistx, y, samd);
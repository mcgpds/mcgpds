function f = mvcgpdsObjective(params, model, samd)

% mvcgpdsObjective Wrapper function for mvcgpds objective.
% FORMAT
% DESC provides a wrapper function for the mvcgpds, it
% takes the negative of the log likelihood, feeding the parameters
% correctly to the model.
% ARG params : the parameters of the mvcgpds model.
% ARG model : the model structure in which the parameters are to be
% placed.
% RETURN f : the negative of the log likelihood of the model.
% 
% SEEALSO : mvcgpdsModelCreate, mvcgpdsLogLikelihood, mvcgpdsExpandParam



model = mvcgpdsExpandParam(model, params, samd);

f = -mvcgpdsLogLikelihood(params, model, samd);

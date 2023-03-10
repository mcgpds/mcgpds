function g = mvcgpdsGradient(params, model, samd)

% mvcgpdsGradient mvcgpds gradient wrapper.
% FORMAT
% DESC is a wrapper function for the gradient of the negative log
% likelihood of a shared variational GP-LVM model with respect to the latent postions
% and parameters.
% ARG params : vector of parameters and latent postions where the
% gradient is to be evaluated.
% ARG model : the model structure into which the latent positions
% and the parameters will be placed.
% RETURN g : the gradient of the negative log likelihood with
% respect to the latent positions and the parameters at the given
% point.
% 
% SEEALSO : mvcgpdsLogLikeGradients, mvcgpdsExpandParam


model = mvcgpdsExpandParam(model, params, samd);
g = -mvcgpdsLogLikeGradients(model, samd);

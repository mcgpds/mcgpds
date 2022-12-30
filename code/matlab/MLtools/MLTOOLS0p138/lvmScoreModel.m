function ll = lvmScoreModel(model)

% LVMSCOREMODEL Score model with a GP log likelihood.
%
%	Description:
%	ll = lvmScoreModel(model)
%% 	lvmScoreModel.m SVN version 1018
% 	last update 2010-10-18T20:34:28.000000Z
  
  options = gpOptions('ftc');
  gpmod = gpCreate(model.q, model.d, model.X, model.Y, options);
  
  gpmod = gpOptimise(gpmod);
  
  ll = modelLogLikelihood(gpmod);
  
end

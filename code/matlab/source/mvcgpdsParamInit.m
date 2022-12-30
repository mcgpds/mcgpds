function model = mvcgpdsParamInit(model,Y,X, samd)
% mvcgpdsParamInit Initialize the mlti-view cgpds from the data
%
%	Description:
%	model = mvcgpdsParamInit(model,Y,X)
%

  
% if input dimension is equal to the latent dimension, 
% then  initialize the variational mean to the normalzed training data
%if model.d == model.q 
%    X = Y; 
%    model.vardist.means = X;  
%    % inducing points 
%    ind = randperm(model.N);
%    ind = ind(1:model.k);
%    model.X_u = X(ind, :);
%end

if ~strcmp(model.kern_u.type,'cmpnd')
   % 
   if strcmp(model.kern_u.type,'rbfard2')  | strcmp(model.kern_u.type,'rbfardjit') 
      % 
       model.kern_u.inputScales = 5./(((max(X)-min(X))).^2);
       %model.kern.variance = max(var(Y)); % !!! better use mean(var(Y)) here...
       model.kern_u.variance = mean(var(Y)); % NEW!!!
      %
   elseif strcmp(model.kern_u.type,'linard2')
      %
      model.kern_u.inputScales = 5./(((max(X)-min(X))).^2);
      %   
   end
   %
else
   %
   for i = 1:length(model.kern_u.comp)
      %
      if strcmp(model.kern_u.comp{i}.type,'rbfard2') | strcmp(model.kern_u.type,'rbfardjit') 
      % 
         model.kern_u.comp{i}.inputScales = 5./(((max(X)-min(X))).^2);
         %model.kern.comp{i}.variance = max(var(Y));
         model.kern_u.comp{i}.variance = var(model.m(:));
      %
      elseif strcmp(model.kern_u.comp{i}.type,'linard2')
      %
         model.kern_u.comp{i}.inputScales = 0.01*max(var(Y))*ones(1,size(X,2));% %5./(((max(X)-min(X))).^2);
      %   
      end
      %
   end
   %
end



if ~strcmp(model.kern_v.type,'cmpnd')
   % 
   if strcmp(model.kern_v.type,'rbfard2')  | strcmp(model.kern_v.type,'rbfardjit') 
      % 
       model.kern_v.inputScales = 5./(((max(X)-min(X))).^2);
       %model.kern.variance = max(var(Y)); % !!! better use mean(var(Y)) here...
       model.kern_v.variance = mean(var(Y)); % NEW!!!
      %
   elseif strcmp(model.kern_v.type,'linard2')
      %
      model.kern_v.inputScales = 5./(((max(X)-min(X))).^2);
      %   
   end
   %
else
   %
   for i = 1:length(model.kern_v.comp)
      %
      if strcmp(model.kern_v.comp{i}.type,'rbfard2') | strcmp(model.kern_v.type,'rbfardjit') 
      % 
         model.kern_v.comp{i}.inputScales = 5./(((max(X)-min(X))).^2);
         %model.kern.comp{i}.variance = max(var(Y));
         model.kern_v.comp{i}.variance = var(model.m(:));
      %
      elseif strcmp(model.kern_v.comp{i}.type,'linard2')
      %
       model.kern_v.comp{i}.inputScales = 0.01*max(var(Y))*ones(1,size(X,2));% %5./(((max(X)-min(X))).^2);
      %   
      end
      %
   end
   %
end


% initialize inducing inputs by kmeans 
%kmeansops = foptions;
%kmeansops(14) = 10;
%kmeansops(5) = 1;
%kmeansops(1) = 0;
%ch = randperm(model.N);
%centres = model.vardist.means(ch(1:model.k),:);
%model.X_u = kmeans(centres, model.vardist.means, kmeansops);

% model.beta = 100;%/max(var(Y));
% model.alpha = rand(1);
% model.epsilon = rand(1);


initParams = cgpdsExtractParam(model, samd);
model.numParams = length(initParams);
% This forces kernel computation.
model = cgpdsExpandParam(model, initParams,samd);


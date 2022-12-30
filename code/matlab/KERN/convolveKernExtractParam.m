function [params, names] = convolveKernExtractParam(kern)


%	Copyright (c) 2013 ZhaoJing
% 	last update 2013-07-20

params = [kern.S' kern.P_d(:)' kern.Lambda_k];
if nargout > 1
  names = {'variance'};
  for i = 1:length(kern.inputScales)
    names{1+i} = ['input scale ' num2str(i)];
  end
end
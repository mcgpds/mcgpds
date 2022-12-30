function y = sigmoidTransformv2(x, transform)

% EXPTRANSFORM Constrains a parameter to be (0,1) through sigmoid.
%
%	Description:
%	y = sigmoidTransform(x, transform)




limVal = 36;
y = zeros(size(x));
eps = 1e-17;
switch transform
 case 'atox'
  index = find(x<-limVal);
  y(index) = eps;
  index = find(x>=-limVal & x<limVal);
  y(index) = 1/(1+exp(-x(index)));
  index = find(x>=limVal);
  y(index) = 1/(1+exp(-limVal));
 case 'xtoa'
  y = -log(1/x-1);
 case 'gradfact'
  y = x*(1-x);
end

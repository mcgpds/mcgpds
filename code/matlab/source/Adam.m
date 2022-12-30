function [x, options, flog, pointlog, scalelog] = Adam(f, x, options, gradf, varargin)
%ADAM optimization.


epslion = 0.001;
rho1 = 0.9;
rho2 = 0.999;
delta = 10e-8;
s = zeros(size(x));
r = zeros(size(x));
t = 0;
model = varargin{1};

%  Set up the options.
if length(options) < 18
  error('Options vector too short')
end

if(options(14))
  niters = options(14);
else
  niters = 100;
end

display = options(1);
gradcheck = options(9);

% Set up strings for evaluating function and gradient
f = fcnchk(f, length(varargin));
gradf = fcnchk(gradf, length(varargin));

nparams = length(x);

%  Check gradients
if (gradcheck)
  feval('gradchek', x, f, gradf, varargin{:});
end

j = 1;

if model.d > 500
    interval = 100;
else
    interval = unidrnd(model.d-1);
end

while (j <= niters)
    % Calculate first and second directional derivatives.
%     t0=cputime;
    start = mod(j,interval);
    if(start == 0)
        start = 1;
    end
    varargin{3} = [start:interval:model.d]; 
%     varargin{3} = [1:model.d]; 
    
    fnow = feval(f, x, varargin{:});
    gradnew = feval(gradf, x, varargin{:});
    t = t + 1;
    s = rho1*s + (1-rho1)*gradnew;
    r = rho2*r + (1-rho2)*gradnew.*gradnew;
    
    shat = s/(1-rho1^t);
    rhat = r/(1-rho2^t);
    dx = epslion*shat./(sqrt(rhat)+delta);
    x = x - dx;
    if display > 0
        fprintf(1, 'Cycle %4d  Error %11.6f Gradient %f\n', j, fnow,gradnew*gradnew');
    end
    j = j + 1;
%     TimeEachIter=cputime-t0;
%     fprintf('TimeEachIter %f\n',TimeEachIter);
end


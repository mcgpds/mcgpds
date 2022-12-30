
% DEMGPTWOSAMPLE Test GP two sample code.
%
%	Description:
%	% 	demGpTwoSample.m SVN version 907
% 	last update 2010-07-23T18:43:38.000000Z

% Generate artificial data. 
numSamps = 3;
tTrue = [1 2 3 4 5 1 2 3 4 5]';
kern = kernCreate(tTrue, 'rbf');
kern.inverseWidth = 1/9;
K = kernCompute(kern, tTrue) + eye(10)*0.01;
yTrue = gsamp(zeros(size(tTrue)), K, numSamps);
tTrue = [1 2 3 4 5]';
K = kernCompute(kern, tTrue) + eye(5)*0.01;
yTrue = [yTrue; gsamp(zeros(1, 10), [K zeros(5); zeros(5) K], numSamps)];

t{1} = [1 2 3 4 5]';
t{2} = [1 2 3 4 5]';
y{1} = yTrue(:, 1:5)';
y{2} = yTrue(:, 6:10)';

llr = gpTwoSample(t, y);
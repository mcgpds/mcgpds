clear 
subsample = 2;
% Select which sequences from the full dataset to read
seqToKeep = [1 3 4 5 6];
% The 8th sequence is used for testing
testSeq = 8;
% How to initialize latent space
initial_X = 'separately';
% How to initialize ARD parameters
latentDimPerModel = [9 6];
% Use dynamical priors

dynUsed=1;

% Optimization settings

initVardistIters = 240;

itNo = [800 700 800 800];

vardistCovarsMult = 0.09;

indPoints = 100;

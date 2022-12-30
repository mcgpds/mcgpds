
% DEMGPTWOSAMPLELIFSH Run GP two sample code on LifSh.
%
%	Description:
%	% 	demGpTwoSampleLifsh.m SVN version 907
% 	last update 2010-07-23T18:44:00.000000Z

% Load data
load shLIF

t{1} = [times_shLif times_shLif]';
y{1} = [c1shLif; c3shLif];
t{2} = [times_shLif times_shLif]';
y{2} = [a7shE13Lif; c4shE13Lif];


[llr, models] = gpTwoSample(t, y);


[void, order] = sort(llr);
for i = order(end:-1:1)'
  fprintf([num2str(llr(i), 4) '\t' genes_vect_shLif{1, i} '\n'])
end

save LIFshresults.mat llr models 


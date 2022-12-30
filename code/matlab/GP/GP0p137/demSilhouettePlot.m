
% DEMSILHOUETTEPLOT
%
%	Description:
%	% 	demSilhouettePlot.m SVN version 1440
% 	last update 2011-06-12T16:40:54.000000Z

% Show prediction for test data.
yPred = modelOut(model, XTest);
xyzankurAnimCompare(yPred, yTest, 96);

yDiff = (yPred - yTest);
rmsError = sqrt(sum(sum(yDiff.*yDiff))/numel(yDiff));

counter = 0;
if printDiagram
  ind = 1:27:size(yPred, 1);
  for i = ind
    counter = counter + 1;
    figure
    handle = xyzankurVisualise(yPred(i,:), 1);
    printPlot([fileBaseName '_' num2str(counter)], '../tex/diagrams', '../html') 
  end
end

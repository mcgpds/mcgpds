function [neighboursInd, A] = findAcyclicNeighbours(Y, k)

% FINDACYCLICNEIGHBOURS find the k nearest neighbours for each point in Y preventing cycles in the graph.
%
%	Description:
%
%	[IND, D] = FINDACYCLICNEIGHBOURS(Y, K) returns the indices of the k
%	nearest neighbours to each point in the given data matrix Y.
%	 Returns:
%	  IND - the indices of each points neighbours.
%	  D - the squared distance to each of the neighbours.
%	 Arguments:
%	  Y - the data in which neighbours need to be found.
%	  K - the number of neighbours that need to be found.
%	
%
%	See also
%	LLEOPTIMISE, FMVUOPTIMISE, ISOMAPCREATE


%	Copyright (c) 2010 Neil D. Lawrence
% 	findAcyclicNeighbours.m SVN version 1414
% 	last update 2010-12-22T18:57:59.000000Z
  
Y2 = sum(Y.*Y, 2);
D2 = repmat(Y2', size(Y, 1), 1) ...
     + repmat(Y2, 1, size(Y, 1)) ...
     - 2*Y*Y';
D2=tril(D2);

D2(find(D2==0)) = inf;
D2(1:size(D2, 1)+1:end) = inf;
[void, ind] = sort(D2);
neighboursInd = ind(1:k, :)';
for i = 1:size(Y, 1) 
  for j = 1:k
    if(isinf(void(j, i)))
      neighboursInd(i, j) = NaN;
    end
  end
end
if nargout > 1
  A = zeros(size(neighboursInd));
  for i = 1:size(A, 1)
    for j = 1:size(A, 2)
      A(i, j) = D2(neighboursInd(i, j), i);
    end
  end
end
    
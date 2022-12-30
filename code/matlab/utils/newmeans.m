function [nmeans,newalpha] = newmeans(dynamics,alpha)
% compute the mean of condinationl gaussian 
% and the 100 new index points for plotting

k = dynamics.kern;
m = dynamics.vardist.means;
told = dynamics.t;

nim_point = 201;
tmp = linspace(-1, 1, nim_point)';
tnew = tmp(1:nim_point, 1); %timeStampsTest = t(indTs,1);
anew = linspace(alpha(1), alpha(end), nim_point);

knm = kernCompute(k,tnew,told);
kmm = kernCompute(k,told);
% kmminv = inv(kmm);
% nm = knm * kmminv * m;
nm = knm * (kmm \ m);

nmeans = nm;
newalpha = tnew;
end


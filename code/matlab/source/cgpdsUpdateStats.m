function model = cgpdsUpdateStats(model, X_v ,X_u, samd)

% VARGPLVMUPDATESTATS Update stats for VARGPLVM model.
%
%	Description:
%	
%	
%
%	See also
%	VARGPLVMOPTIMISE, VARGPLVMEXPANDPARAM


%	Copyright (c) 2009-2011 Michalis K. Titsias
%	Copyright (c) 2009-2011 Neil D. Lawrence
%	Copyright (c) 2010-2011 Andreas C. Damianou
% 	vargplvmUpdateStats.m SVN version 1460
% 	last update 2011-07-04T19:19:55.344562Z   
  
jitter = 1e-6;
lend = length(samd);
%model.jitter = 1e-6;


% %%% Precomputations for the KL term %%%
%%% Maybe we should add something like model.dynamics.X =
%%% model.dynamics.vardist.means (if visualisation requires that).
 if isfield(model, 'dynamics') && ~isempty(model.dynamics)
     model = vargplvmDynamicsUpdateStats(model);
 end

%%% Precomputations for (the likelihood term of) the bound %%%

K_uu = kernCompute(model.kern_u, X_u);
if isempty(find(isnan(K_uu)))&&isempty(find(isinf(K_uu)))
    model.K_uu = K_uu;%M*M
end
K_vv = kernCompute(model.kern_v, X_v);
if isempty(find(isnan(K_vv)))&&isempty(find(isinf(K_vv)))
    model.K_vv = K_vv;%M*M
end

% model.K_uu = kernCompute(model.kern_u, X_u);%M*M
% model.K_vv = kernCompute(model.kern_v, X_v);%M*M


% Always add jitter (so that the inducing variables are "jitter" function variables)
% and the above value represents the minimum jitter value
% Putting jitter always ("if" in comments) is like having a second
% whiteVariance in the kernel which is constant.
%if (~isfield(model.kern, 'whiteVariance')) | model.kern.whiteVariance < jitter
   % There is no white noise term so add some jitter.
if ~strcmp(model.kern_u.type, 'rbfardjit')
       model.K_uu = model.K_uu ...
                         + sparseDiag(repmat(jitter, size(model.K_uu, 1), 1));
end
%end

if ~strcmp(model.kern_v.type, 'rbfardjit')
       model.K_vv = model.K_vv ...
                         + sparseDiag(repmat(jitter, size(model.K_vv, 1), 1));
end


model.Psi0 = kernVardistPsi1Compute(model.kern_v, model.vardist, X_v);%N*M
model.Psi1 = kernVardistPsi1Compute(model.kern_u, model.vardist, X_u);%N*M

model.Psi2 = kernVardistPsi0Compute(model.kern_v, model.vardist);%1*1
model.Psi3 = kernVardistPsi0Compute(model.kern_u, model.vardist);%1*1

[model.Psi4, AS_v] = kernVardistPsi2Compute(model.kern_v, model.vardist, X_v);%M*M
[model.Psi5, AS_u] = kernVardistPsi2Compute(model.kern_u, model.vardist, X_u);%M*M

model.Lm_u = jitChol(model.K_uu)';      % M x M: L_m (lower triangular)   ---- O(m^3)
model.invLm_u = model.Lm_u\eye(model.k);  % M x M: L_m^{-1}                 ---- O(m^3)
model.invLmT_u = model.invLm_u'; % L_m^{-T}
model.invK_uu = model.invLmT_u * model.invLm_u;

model.Lm_v = jitChol(model.K_vv)';      % M x M: L_m (lower triangular)   ---- O(m^3)
model.invLm_v = model.Lm_v\eye(model.k);  % M x M: L_m^{-1}                 ---- O(m^3)
model.invLmT_v = model.invLm_v'; % L_m^{-T}
model.invK_vv = model.invLmT_v * model.invLm_v;

if isfield(model,'dynamics')&&~isempty(model.dynamics)
    model.X = model.dynamics.vardist.means;
else
    model.X = model.vardist.means;
end




%K_uu_jit = model.K_uu + model.jitter*eye(model.k);
%model.Lm = chol(K_uu_jit, 'lower');          
  
% global isFirst;
% 
% if isFirst == 1
%     model.C_u = cell(model.d,model.J);
%     model.TrC_u = zeros(model.d,model.J);
%     model.At_u = cell(model.d,model.J);
%     model.Lat_u = cell(model.d,model.J);
%     model.invLat_u = cell(model.d,model.J);
%     model.invLatT_u = cell(model.d,model.J);
%     model.logDetAt_u = zeros(model.d,model.J);
%     model.P1_u = cell(model.d,model.J);
%     model.P_u = cell(model.d,model.J);
%     model.TrPP_u = 0;
%     model.TrPP_u_temp = zeros(model.d,model.J);
%     model.B_u = cell(model.d,model.J);
%     model.T1_u = zeros(model.k,model.k);
%     model.T1_u_temp = cell(model.d,model.J);
% end



% model.C_u_temp = model.invLm_u * model.Psi5 * model.invLmT_u;  %M*M
% 
% for dd = 1:lend
%     d = samd(dd);
%     for j = 1:model.J
%         %model.Lm = chol(model.K_uu, 'lower'); 
%         
%         model.C_u{d,j} = model.W(d,j)^2 * model.C_u_temp;
%         model.TrC_u(d,j) = sum(diag(model.C_u{d,j})); % Tr(C)
%         % Matrix At replaces the matrix A of the old implementation; At is more stable
%         % since it has a much smaller condition number than A=sigma^2 K_uu + Psi2
%         model.At_u{d,j} = (1/model.beta) * eye(size(model.C_u{d,j},1)) + model.C_u{d,j}; % At = beta^{-1} I + C
%         model.Lat_u{d,j} = jitChol(model.At_u{d,j})';%lower bound
%         model.invLat_u{d,j} = model.Lat_u{d,j}\eye(size(model.Lat_u{d,j},1));  
%         model.invLatT_u{d,j} = model.invLat_u{d,j}';
%         model.logDetAt_u(d,j) = 2*(sum(log(diag(model.Lat_u{d,j})))); % log |At|
% 
%         model.P1_u{d,j} = model.invLat_u{d,j} * model.invLm_u; % M x M
% 
%         % First multiply the two last factors; so, the large N is only involved
%         % once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
%         model.P_u{d,j} = model.P1_u{d,j} * (model.Psi1' * model.m(:,d));%M*1  
% 
%         % Needed for both, the bound's and the derivs. calculations.
% %         model.TrPP_u = sum(sum(model.P_u .* model.P_u));
%         model.TrPP_u_temp(d,j) = model.W(d,j)^2*model.P_u{d,j}'*model.P_u{d,j};%1*1
%         model.B_u{d,j} = model.P1_u{d,j}' * model.P_u{d,j}; %M*1
%         
%         if isFirst == 0
%             model.T1_u = model.T1_u - model.T1_u_temp{d,j};
%         end
%         Tb_u = (1/model.beta) * model.W(d,j)^2 * (model.P1_u{d,j}' * model.P1_u{d,j})...
%         	 + model.W(d,j)^4 * model.B_u{d,j} * model.B_u{d,j}';%M*M
%         model.T1_u_temp{d,j} = model.W(d,j)^2 * model.invK_uu - Tb_u; % sum w.r.t all D and all J
%         model.T1_u = model.T1_u + model.T1_u_temp{d,j};
%     end
% end
% 
% model.TrPP_u = sum(sum(model.TrPP_u_temp));



%model.Lm_v = chol(model.K_uu, 'lower'); 

C_v = model.invLm_v * model.Psi4 * model.invLmT_v;
if isempty(find(isnan(C_v)))&&isempty(find(isinf(C_v)))
    model.C_v = C_v;
end
% model.C_v = model.invLm_v * model.Psi4 * model.invLmT_v;

model.TrC_v = sum(diag(model.C_v)); % Tr(C)
% Matrix At replaces the matrix A of the old implementation; At is more stable
% since it has a much smaller condition number than A=sigma^2 K_uu + Psi2

model.At_v = (1/model.beta) * eye(size(model.C_v,1)) + model.C_v; % At = beta^{-1} I + C
model.Lat_v = jitChol(model.At_v)';
model.invLat_v = model.Lat_v\eye(size(model.Lat_v,1));  
model.invLatT_v = model.invLat_v';
model.logDetAt_v = 2*(sum(log(diag(model.Lat_v)))); % log |At|

model.P1_v = model.invLat_v * model.invLm_v; % M x M

% First multiply the two last factors; so, the large N is only involved
% once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
model.P_v = model.P1_v * (model.Psi0' * model.m);%M*D

% Needed for both, the bound's and the derivs. calculations.
model.TrPP_v = sum(sum(model.P_v .* model.P_v));%1*1


%%% Precomputations for the derivatives (of the likelihood term) of the bound %%%

%model.B = model.invLmT * model.invLatT * model.P; %next line is better
model.B_v = model.P1_v' * model.P_v;%M*D

Tb_v = (1/model.beta) * model.d * (model.P1_v' * model.P1_v);
	Tb_v = Tb_v + (model.B_v * model.B_v');
model.T1_v = model.d * model.invK_vv - Tb_v;



%{
%%%%-------------------------- TEMP (for analysing how ind. pts are optimised) !!!!!!!!!
d=dist2(model.X_u, model.vardist.means);
[C,I]=min(d,[],2);
dNorm=dist2(model.X_u, model.vardist.means)/norm(model.X_u)^2;
[Cnorm,Inorm]=min(dNorm,[],2);

if Inorm-I ~= 0
    fprintf(1, 'Something is wrong with the distances! (Update Stats)');
end

try 
    load TEMPInducingMeansDist
catch exception
    TEMPInducingMeansDist=[];
end
try 
    load TEMPInducingIndices
catch exception
    TEMPInducingIndices=[];
end
try 
    load TEMPInducingMeansDistNorm
catch exception
    TEMPInducingMeansDistNorm=[];
end
TEMPInducingMeansDist = [TEMPInducingMeansDist sum(C)];
save 'TEMPInducingMeansDist.mat' 'TEMPInducingMeansDist';
%fprintf(1,'# Total X_u - vardistMeans = %d\n',sum(C));
TEMPInducingIndices = [TEMPInducingIndices I];
save 'TEMPInducingIndices.mat' 'TEMPInducingIndices';

TEMPInducingMeansDistNorm = [TEMPInducingMeansDistNorm sum(Cnorm)];
save 'TEMPInducingMeansDistNorm.mat' 'TEMPInducingMeansDistNorm';
%}




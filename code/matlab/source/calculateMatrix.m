function tempMatrix = calculateMatrix( model,samd)

    lend = length(samd);

    % calculate matrixs for the computation of elbo and gradient

        tempMatrix.C_u = cell(lend,model.J);
        tempMatrix.TrC_u = zeros(lend,model.J);
        tempMatrix.At_u = cell(lend,model.J);
        tempMatrix.Lat_u = cell(lend,model.J);
        tempMatrix.invLat_u = cell(lend,model.J);
        tempMatrix.invLatT_u = cell(lend,model.J);
        tempMatrix.logDetAt_u = zeros(lend,model.J);
        tempMatrix.P1_u = cell(lend,model.J);
        tempMatrix.P_u = cell(lend,model.J);
        tempMatrix.TrPP_u = 0;
        TrPP_u = zeros(lend,model.J);
        tempMatrix.B_u = cell(lend,model.J);
        tempMatrix.T1_u = zeros(model.k,model.k);
        

        tempMatrix.C_u_temp = model.invLm_u * model.Psi5 * model.invLmT_u;  %M*M
        if isempty(find(isnan(tempMatrix.C_u_temp)))&&isempty(find(isinf(tempMatrix.C_u_temp)))
            
        else
            n = size(tempMatrix.C_u_temp,1);
            X = diag(rand(n,1));
            U = orth(rand(n,n));
            tempMatrix.C_u_temp = U'*X*U;
        end
        
        if ~(isfield(model,'fixU') && model.fixU == 1)
            for d = 1:lend
                dd = samd(d);
                for j = 1:model.J
                    %model.Lm = chol(model.K_uu, 'lower');
                    
                    tempMatrix.C_u{d,j} = model.W(dd,j)^2 * tempMatrix.C_u_temp;
                    tempMatrix.TrC_u(d,j) = sum(diag(tempMatrix.C_u{d,j})); % Tr(C)
                    % Matrix At replaces the matrix A of the old implementation; At is more stable
                    % since it has a much smaller condition number than A=sigma^2 K_uu + Psi2
                    tempMatrix.At_u{d,j} = (1/model.beta) * eye(size(tempMatrix.C_u{d,j},1)) + tempMatrix.C_u{d,j}; % At = beta^{-1} I + C
                    tempMatrix.Lat_u{d,j} = jitChol(tempMatrix.At_u{d,j})';%lower bound
                    tempMatrix.invLat_u{d,j} = tempMatrix.Lat_u{d,j}\eye(size(tempMatrix.Lat_u{d,j},1));
                    tempMatrix.invLatT_u{d,j} = tempMatrix.invLat_u{d,j}';
                    tempMatrix.logDetAt_u(d,j) = 2*(sum(log(diag(tempMatrix.Lat_u{d,j})))); % log |At|
                    
                    tempMatrix.P1_u{d,j} = tempMatrix.invLat_u{d,j} * model.invLm_u; % M x M
                    
                    % First multiply the two last factors; so, the large N is only involved
                    % once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
                    tempMatrix.P_u{d,j} = tempMatrix.P1_u{d,j} * (model.Psi1' * model.m(:,dd));%M*1
                    
                    % Needed for both, the bound's and the derivs. calculations.
                    %         model.TrPP_u = sum(sum(P_u .* P_u));
                    TrPP_u(d,j) = model.W(dd,j)^2*tempMatrix.P_u{d,j}'*tempMatrix.P_u{d,j};%1*1
                    tempMatrix.B_u{d,j} =tempMatrix.P1_u{d,j}' * tempMatrix.P_u{d,j}; %M*1
                    
                    
                    Tb_u = (1/model.beta) * model.W(dd,j)^2 * (tempMatrix.P1_u{d,j}' * tempMatrix.P1_u{d,j})...
                        + model.W(dd,j)^4 * tempMatrix.B_u{d,j} * tempMatrix.B_u{d,j}';%M*M
                    tempMatrix.T1_u = tempMatrix.T1_u + model.W(dd,j)^2 * model.invK_uu - Tb_u;
                end
            end
        end
        
        
        tempMatrix.TrYYdr = trace(model.m(:,samd)*model.m(:,samd)');

        tempMatrix.TrPP_u = sum(sum(TrPP_u));

        tempMatrix.C_v = model.invLm_v * model.Psi4 * model.invLmT_v;
        if isempty(find(isnan(tempMatrix.C_v)))&&isempty(find(isinf(tempMatrix.C_v)))
        else
            n = size(tempMatrix.C_v,1);
            X = diag(rand(n,1));
            U = orth(rand(n,n));
            tempMatrix.C_v = U'*X*U;
        end
        
        tempMatrix.TrC_v = sum(diag(tempMatrix.C_v)); % Tr(C)
        % Matrix At replaces the matrix A of the old implementation; At is more stable
        % since it has a much smaller condition number than A=sigma^2 K_uu + Psi2
        tempMatrix.At_v = (1/model.beta) * eye(size(tempMatrix.C_v,1)) + tempMatrix.C_v; % At = beta^{-1} I + C
        tempMatrix.Lat_v = jitChol(tempMatrix.At_v)';
        tempMatrix.invLat_v = tempMatrix.Lat_v\eye(size(tempMatrix.Lat_v,1));  
        tempMatrix.invLatT_v = tempMatrix.invLat_v';
        tempMatrix.logDetAt_v = 2*(sum(log(diag(tempMatrix.Lat_v)))); % log |At|

        tempMatrix.P1_v = tempMatrix.invLat_v * model.invLm_v; % M x M

        % First multiply the two last factors; so, the large N is only involved
        % once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
        tempMatrix.P_v = tempMatrix.P1_v * (model.Psi0' * model.m(:,samd));%M*D

        % Needed for both, the bound's and the derivs. calculations.
        tempMatrix.TrPP_v = sum(sum(tempMatrix.P_v .* tempMatrix.P_v));%1*1


        %%% Precomputations for the derivatives (of the likelihood term) of the bound %%%

        %model.B = model.invLmT * model.invLatT * model.P; %next line is better
        tempMatrix.B_v = tempMatrix.P1_v' * tempMatrix.P_v;%M*D

        Tb_v = (1/model.beta) * lend* (tempMatrix.P1_v' * tempMatrix.P1_v);
            Tb_v = Tb_v + (tempMatrix.B_v * tempMatrix.B_v');
        tempMatrix.T1_v = lend* model.invK_vv - Tb_v;
end


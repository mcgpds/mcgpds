function [ A] = positiveDefiniteMatrix( A )
%确保矩阵的正定性(输入的A必须是方阵)
%输入的矩阵A可能不正定，通过一些操作使矩阵正定
%inA是矩阵A的逆矩阵

            A=(A+A')/2;%对称阵
            n=size(A,1);%A
           [v,l]= eig(A);%返回矩阵的特征向量v（每一列对应于某个特征值的特征向量），l是特征值，是一个对角阵，对角线元素为特征值。
            l(find(l<0))=1e-10;%将小于0的特征值赋值为0
            A=v*l*v';%得到半正定矩阵
            eps = 10^-5;
            A=A+eye(n)*eps;% 正定矩阵
            
%             A=(A+A')/2;
%             
%             [v1,l1]=eig(A);
%             inA1=inv(A);
%            c = chol(A);
%            inA2 = inv(c)*inv(c)';%B1的逆矩阵
%             inA3 = matrixinverse( A );
            
            
            
            
%             ll(find(ll<eps))=eps;%小于eps的特征值被赋值为eps,负数也被赋值为0.
%             ll = diag(diag(ll));
%             A = v*ll*v'+eye(n)*eps;%循环 直到矩阵KK可逆，也就是满秩。
%             dd= det(A);
%             while dd==0
%                 A=(A+A')/2;
%                 [v,l]= eig(A);
%                 ll  = l;
%                 eps = eps*10;
%                 ll(find(ll<eps))=eps;
%                 ll = diag(diag(ll));
%                 A = v*ll*v'+eye(n)*eps;
%                 dd= det(A);
%             end
%             A=(A+A')/2;
%             c = chol(A);
%             inA = inv(c)*inv(c)';%B1的逆矩阵
            %lndetA1{j,r}=2 * sum(log(diag(c))); 
end


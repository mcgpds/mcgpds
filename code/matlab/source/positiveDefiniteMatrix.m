function [ A] = positiveDefiniteMatrix( A )
%ȷ�������������(�����A�����Ƿ���)
%����ľ���A���ܲ�������ͨ��һЩ����ʹ��������
%inA�Ǿ���A�������

            A=(A+A')/2;%�Գ���
            n=size(A,1);%A
           [v,l]= eig(A);%���ؾ������������v��ÿһ�ж�Ӧ��ĳ������ֵ��������������l������ֵ����һ���Խ��󣬶Խ���Ԫ��Ϊ����ֵ��
            l(find(l<0))=1e-10;%��С��0������ֵ��ֵΪ0
            A=v*l*v';%�õ�����������
            eps = 10^-5;
            A=A+eye(n)*eps;% ��������
            
%             A=(A+A')/2;
%             
%             [v1,l1]=eig(A);
%             inA1=inv(A);
%            c = chol(A);
%            inA2 = inv(c)*inv(c)';%B1�������
%             inA3 = matrixinverse( A );
            
            
            
            
%             ll(find(ll<eps))=eps;%С��eps������ֵ����ֵΪeps,����Ҳ����ֵΪ0.
%             ll = diag(diag(ll));
%             A = v*ll*v'+eye(n)*eps;%ѭ�� ֱ������KK���棬Ҳ�������ȡ�
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
%             inA = inv(c)*inv(c)';%B1�������
            %lndetA1{j,r}=2 * sum(log(diag(c))); 
end


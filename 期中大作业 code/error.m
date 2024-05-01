function [L1_error_DRP,L1_error_DRP_M,L1_error_MDCD,L1_error_SA_DRP] = L1_error(solve,N,m)
    L1_error_DRP=zeros(1,5);
    L1_error_DRP_M=zeros(1,5);
    L1_error_MDCD=zeros(1,5);
    L1_error_SA_DRP=zeros(1,5);
    for i=1:length(N)
        [x,u_Exact,u_DRP,u_DRP_M,u_MDCD,u_SA_DRP]=solve(N(i),m);
        dx=x(2)-x(1);
        L1_error_DRP(i) = dx*sum(abs(u_Exact-u_DRP));
        L1_error_DRP_M(i) = dx*sum(abs(u_Exact-u_DRP_M));
        L1_error_MDCD(i) = dx*sum(abs(u_Exact-u_MDCD));
        L1_error_SA_DRP(i) = dx*sum(abs(u_Exact-u_SA_DRP));
    end
    end
    
function [L2_error_DRP,L2_error_DRP_M,L2_error_MDCD,L2_error_SA_DRP] = L2_error(solve,N,m)
    L2_error_DRP=zeros(1,5);
    L2_error_DRP_M=zeros(1,5);
    L2_error_MDCD=zeros(1,5);
    L2_error_SA_DRP=zeros(1,5);
    for i=1:length(N)
        [x,u_Exact,u_DRP,u_DRP_M,u_MDCD,u_SA_DRP]=solve(N(i),m);
        dx=x(2)-x(1);
        L2_error_DRP(i) = (dx*sum((u_Exact-u_DRP).^2))^0.5;
        L2_error_DRP_M(i) = (dx*sum((u_Exact-u_DRP_M).^2))^0.5;
        L2_error_MDCD(i) = (dx*sum((u_Exact-u_MDCD).^2))^0.5;
        L2_error_SA_DRP(i) = (dx*sum((u_Exact-u_SA_DRP).^2))^0.5;
    end
    end
    
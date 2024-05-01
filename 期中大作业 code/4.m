clear;

N=128;
[x,u_Exact,u_DRP,u_DRP_M,u_MDCD,u_SA_DRP]=solve4(N);

N2=2.^(5:9);
[L1_error_DRP,L1_error_DRP_M,L1_error_MDCD,L1_error_SA_DRP] = L1_error(@solve4,N2,5);

figure(1)
plot(x,u_Exact,'k','LineWidth',1)
hold on
plot(x,u_DRP,'sr','LineWidth',0.7, 'MarkerSize', 4.5, 'MarkerFaceColor', 'w')
hold on
plot(x,u_DRP_M,'v','Color',[1,0.5,0],'LineWidth',0.7, 'MarkerSize', 4.5, 'MarkerFaceColor', 'w')
hold on
plot(x,u_MDCD,'^g','LineWidth',0.7, 'MarkerSize', 4.5, 'MarkerFaceColor', 'w')
hold on
plot(x,u_SA_DRP,'ob','LineWidth',0.7, 'MarkerSize', 4.5, 'MarkerFaceColor', 'w')
hold off

set(gca,'linewidth',1)
legend('Exact','DRP','DRP-M','MDCD','SA-DRP','Location','northwest')

figure(4)
loglog(N2,L1_error_DRP,'-sr','LineWidth',1)
hold on
loglog(N2,L1_error_DRP_M,'-v','Color',[1,0.5,0],'LineWidth',1)
hold on
loglog(N2,L1_error_MDCD,'-^g','LineWidth',1)
hold on
loglog(N2,L1_error_SA_DRP,'-ob','LineWidth',1)
hold off

set(gca,'linewidth',0.7)
legend('DRP','DRP-M','MDCD','SA-DRP')
xlim([2e1, 9e2])
ylim([1e-6,1e0])
xlabel('N')
ylabel('L1 error')

function [x,u_Exact,u_DRP,u_DRP_M,u_MDCD,u_SA_DRP]=solve4(N,~)
a=1;
CFL=0.2;
xRange = [0, 1];
dx=(xRange(1,2)-xRange(1,1))/N;
x=xRange(1,1):dx:xRange(1,2);
dt=CFL*dx;
t0=0;
tf=1.0;
tf=floor(tf/dt)*dt;

phi=rand(64,1);
epsilon=0.1;

u0=ones(1,N+1);
for k=1:64
    u0=u0+epsilon*(k/12)^2*exp(-(k/12)^2).*sin(2*pi*k.*(x+phi(k)));
end
u_Exact=ones(1,N+1);
for k=1:64
    u_Exact=u_Exact+epsilon*(k/12)^2*exp(-(k/12)^2).*sin(2*pi*k.*(x-tf+phi(k)));
end

u_DRP=u0;
for time=(t0+dt):dt:tf
    u_DRP=Runge_Kutta4(@DRP_scheme,a,u_DRP,dt,dx);
end

u_DRP_M=u0;
for time=(t0+dt):dt:tf
    u_DRP_M=Runge_Kutta4(@DRP_M_scheme,a,u_DRP_M,dt,dx);
end

u_MDCD=u0;
for time=(t0+dt):dt:tf
    u_MDCD=Runge_Kutta4(@MDCD_scheme,a,u_MDCD,dt,dx);
end

u_SA_DRP=u0;
for time=(t0+dt):dt:tf
    u_SA_DRP=Runge_Kutta4(@SA_DRP_scheme,a,u_SA_DRP,dt,dx);
end

end

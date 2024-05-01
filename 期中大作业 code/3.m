clear;

N=256;
m=20;
a=1;
CFL=0.3;
xRange = [0, 1];
dx=(xRange(1,2)-xRange(1,1))/N;
x=xRange(1,1):dx:xRange(1,2);
dt=CFL*dx;
t0=0;
tf=10.0;
tf=floor(tf/dt)*dt;

u0=zeros(1,N+1);
for l=1:m
    u0=u0+1/m.*sin(2*pi*l.*x);
end
u_Exact=zeros(1,N+1);
for l=1:m
    u_Exact=u_Exact+1/m.*sin(2*pi*l.*(x-tf));
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

N2=2.^(6:10);
[L2_error_DRP,L2_error_DRP_M,L2_error_MDCD,L2_error_SA_DRP] = L2_error(@solve3,N2,5);

figure(1)
plot(x,u_Exact,'k','LineWidth', 1)
hold on
plot(x,u_DRP,'sr','LineWidth',0.7, 'MarkerSize', 4)
hold on
plot(x,u_DRP_M,'v','Color',[1,0.5,0],'LineWidth',0.7, 'MarkerSize', 4)
hold on
plot(x,u_MDCD,'^g','LineWidth',0.7, 'MarkerSize', 4)
hold on
plot(x,u_SA_DRP,'ob','LineWidth',0.7, 'MarkerSize', 4)
hold off

set(gca,'linewidth',1)
legend('Exact','DRP','DRP-M','MDCD','SA-DRP')
xlim([0.8,1])
ylim([-1,0.4])
xlabel('x')
ylabel('u')

figure(4)
loglog(N2,L2_error_DRP,'-sr','LineWidth',1)
hold on
loglog(N2,L2_error_DRP_M,'-v','Color',[1,0.5,0],'LineWidth',1)
hold on
loglog(N2,L2_error_MDCD,'-^g','LineWidth',1)
hold on
loglog(N2,L2_error_SA_DRP,'-ob','LineWidth',1)
hold off

set(gca,'linewidth',0.7)
legend('DRP','DRP-M','MDCD','SA-DRP')
ylim([1e-9,1e2])
xlabel('N')
ylabel('L2 error')

function [x,u_Exact,u_DRP,u_DRP_M,u_MDCD,u_SA_DRP]=solve3(N,m)
a=1;
CFL=0.3;
xRange = [0, 1];
dx=(xRange(1,2)-xRange(1,1))/N;
x=xRange(1,1):dx:xRange(1,2);
dt=CFL*dx;
t0=0;
tf=10.0;
tf=floor(tf/dt)*dt;

u0=zeros(1,N+1);
for l=1:m
    u0=u0+1/m.*sin(2*pi*l.*x);
end
u_Exact=zeros(1,N+1);
for l=1:m
    u_Exact=u_Exact+1/m.*sin(2*pi*l.*(x-tf));
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
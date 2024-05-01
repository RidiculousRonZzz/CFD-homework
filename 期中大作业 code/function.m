function partial_u_partial_x = DRP_M_scheme(u, dx, ~)
a_j=[-0.020843142770 0.166705904415 -0.770882380518 0 0.770882380518 -0.166705904415 0.020843142770];
a_j=a_j./dx;
partial_u_partial_x=zeros(1,length(u));
for j=-3:0
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(end+j):(end-1),1:(end+j)]);
end
for j=1:3
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(1+j):end,2:(1+j)]);
end
end

function partial_u_partial_x = DRP_scheme(u, dx, ~)
a_j=[-0.02651995 0.18941314 -0.79926643 0 0.79926643 -0.18941314 0.02651995];
a_j=a_j./dx;
partial_u_partial_x=zeros(1,length(u));
for j=-3:0
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(end+j):(end-1),1:(end+j)]);
end
for j=1:3
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(1+j):end,2:(1+j)]);
end
end

function partial_u_partial_x = MDCD_scheme(u, dx, ~)
gama_disp=0.0463783;
gama_diss=0.001;
a_j=[(-1/2*gama_disp-1/2*gama_diss), (2*gama_disp+3*gama_diss+1/12),...
    (-5/2*gama_disp-15/2*gama_diss-2/3), 10*gama_diss, (5/2*gama_disp-15/2*gama_diss+2/3),...
    (-2*gama_disp+3*gama_diss-1/12), (1/2*gama_disp-1/2*gama_diss)];
a_j=a_j./dx;
partial_u_partial_x=zeros(1,length(u));
for j=-3:0
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(end+j):(end-1),1:(end+j)]);
end
for j=1:3
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(1+j):end,2:(1+j)]);
end
end

function partial_u_partial_x = SA_DRP_scheme(u, dx, a)
S1=u([2:end,2])-2*u+u([end-1,1:(end-1)]);
S2=(u([3:end,2:3])-2*u+u([(end-2):(end-1),1:(end-2)]))/4;
S3=u([3:end,2:3])-2*u([2:end,2])+u;
S4=(u([4:end,2:4])-2*u([2:end,2])+u([end-1,1:(end-1)]))/4;
C1=u([2:end,2])-u;
C2=(u([3:end,2:3])-u([end-1,1:(end-1)]))/3;
tempSa=abs(S1+S2);
tempSb=abs(S1-S2);
tempSc=abs(S3+S4);
tempSd=abs(S3-S4);
tempCa=abs(C1+C2);
tempCb=abs(C1-C2);
epsilon=10e-8;
k_ESW_h=acos(2.*min((abs(tempSa-tempSb) + abs(tempSc-tempSd) + abs(tempCa-tempCb./2) + 2*epsilon) ...
    ./(tempSa+tempSb + tempSc+tempSd + tempCa+tempCb + epsilon),1)-1);
gama_disp_h=1/30.*((0<=k_ESW_h)&(k_ESW_h<0.01))+...
    (k_ESW_h+1/6.*sin(2.*k_ESW_h)-4/3*sin(k_ESW_h))./...
    (sin(3.*k_ESW_h)-4.*sin(2.*k_ESW_h)+5.*sin(k_ESW_h)).*((0.01<=k_ESW_h)&(k_ESW_h<2.5))+...
    0.1985842.*((0>k_ESW_h)|(k_ESW_h>=2.5));
gama_diss_h=sign(a).*(0.001.*((0<=k_ESW_h)&(k_ESW_h<1.0))+...
    min(0.001+0.011*((k_ESW_h-1)./(pi-1)).^0.5,0.012).*((0>k_ESW_h)|(k_ESW_h>=1.0)));
a_j_h=[(1/2*gama_disp_h+1/2*gama_diss_h); (-3/2*gama_disp_h-5/2*gama_diss_h-1/12);...
    (gama_disp_h+5*gama_diss_h+7/12); (gama_disp_h-5*gama_diss_h+7/12);...
    (-3/2*gama_disp_h+5/2*gama_diss_h-1/12); (1/2*gama_disp_h-1/2*gama_diss_h)];
a_j_h=a_j_h./dx;
leng=length(u);
a_j=[zeros(1,leng);a_j_h]-[a_j_h(:,[end-1,1:(end-1)]);zeros(1,leng)];
partial_u_partial_x=zeros(1,length(u));
for j=-3:0
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(end+j):(end-1),1:(end+j)]);
end
for j=1:3
    partial_u_partial_x=partial_u_partial_x+a_j(j+4).*u([(1+j):end,2:(1+j)]);
end
end

function u_next = Runge_Kutta4(x_scheme, a, u, dt, dx)
u1=u-dt./4.*a.*x_scheme(u,dx,a);
u2=u-dt./3.*a.*x_scheme(u1,dx,a);
u3=u-dt./2.*a.*x_scheme(u2,dx,a);
u_next=u-dt./1.*a.*x_scheme(u3,dx,a);
end
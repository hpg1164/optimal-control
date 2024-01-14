%% single shotting method
addpath('C:\Users\User\OneDrive\Desktop\casadi')
import casadi.*
T = 0.2;  %sampling time
N = 10;  %perdiction horizon
rob_dia = 0.3;

v_max = 0.6; 
v_min = -v_max;
omega_max = pi/4;
omega_min = -omega_max;
x = SX.sym('x');
y = SX.sym('y');
theta = SX.sym('theta');
states = [x;y;theta];
n_states = length(states);
controls = [v;omega];
n_controls = length(controls);
rhs = [v*cos(theta);v*sin(theta);omega] ; %system rhs

f = Function('f',{states,controls},{rhs});  %non linear mappiing

U = SX.sym('U',n_controls,N);  % controls
P = SX.sym('p',n_states + n_states);  % parameter which include initial and final reference

X = SX.sym('X',n_states,(N+1));


%%compute solution symbolically
X(:,1) = P(1:3);
for k = 1:N
    st = X(:,k);
    con = U(:,k);
    f_value = f(st,con);
    st_next = st+(T*f_value);
    X(:,k+1) = st_next;
end
%this function get optimal trajectory
ff = Function('ff',{U,P},{X});
  obj = 0;  % objective function
  g = []; % constraints vector


  Q = zeros(3,3);Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1;%weighing matraces(states)
  R = zeros(2,2);R(1,1) = 0.5; R(2,2) = 0.05;  %controls
  %compute objective
  for k = k-1:N
      st = X(:,k); con = U(:,k);
      obj = obj+(st-P(4:6))'*Q*(st-P(4:6))+con'*R*con;
  end
  %compute constraints
  for k = 1:N+1
      g = [g ; X(1,k)];  %state x
      g = [g ; X(2,k)];   %state y
  end



  % make the decision variable one column vector
  OPT_variables = reshape(U,2*N,1);
  nlp_prob = struct('f' ,obj,'x' ,OPT_variables,'g',g,'p',P);
  opts = struct;
  opts.ipopt.max_iter = 100;
  opts.ipopt.print_level = 0;
  opts.print_time = 0;
  opts.ipopt.acceptable_tol= 1e-8;
  opts.ipopt.acceptable_obj_change_tol = 1e-6;


  solver = nlpsol('solver','ipopt',nlp_prob,opts);



  args = struct;
  %inequality constraints
  args.lbg = -2; %lower bound
  args.ubg = 2;  %upper bound
  % input conatraints
  args.lbx(1:2:2*N-1,1) = v_min; args.lbx(2:2:2*N,1) = omega_min;
  args.ubx(1:2:2*N-1,1) = v_max ;args. ubx(2:2:2*N,1) = omega_max;


  %%simulation start from here
  t0 = 0;
  x0 = [0;0;0.0];  %initial condition
  xs = [1.5;1.5;0.0];% reference
  xx(:,1) = x0;
  u0 = zeros(N,2);
  sim_tim = 20;


  %%start mpc
  mpciter = 0;
  xx1 = [];
  u_cl = [];

  while(norm((x0-xs),2)> 1e-2 && mpciter < sim_tim/T)
      args.p = [x0;xs];
      args.x0 = reshape(u0',2*N,1);
      sol = solver('x0',args.x0,'lbx',args.lbx,'ubx',args.ubx,'lbg', args.lbg,'ubg',args.ubg,'p',args.p);
      u = reshape(full(sol.x)',2,N)';
      ff_value = ff(u', args.p);

      xx1(:, 1:3, mpciter+1) = full(ff_value)';

      u_cl = [u_cl;u(1,:)];
      t(mpciter+1) = t0;
     
       [t0,x0,u0] = shift(T,t0,x0,u,f)
     
      xx(:,mpciter+2) = x0;
      mpciter = mpciter +1;



  end;
  Draw_MPC_point_stabilization_v1 (t,xx,xx1,u_cl,xs,N,rob_dia)


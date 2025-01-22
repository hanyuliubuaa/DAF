function y_dot = Lorenz(t,y)
% Lorenz Model
% In: Conjugate_Unscented_Transformation_Applications_to_Estimation_and_Control
sigma = 10;
rho = 28;
beta = 8/3;
nx = 3;

y_dot = zeros(nx,1);
x1 = y(1);
x2 = y(2);
x3 = y(3);
x1_dot = sigma*(x2-x1);
x2_dot = rho*x1-x2-x1*x3;
x3_dot = x1*x2-beta*x3;
y_dot(1:nx,1) = [x1_dot;x2_dot;x3_dot];

end
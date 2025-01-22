function x = integ_Lorenz(x0,tspan)
% RK integ on Lorenz Model
opts = odeset('Reltol',1e-13,'AbsTol',1e-13);

y0 = x0;

[t,y] = ode113(@Lorenz, tspan, y0, opts);

x = y(end,:)';
end
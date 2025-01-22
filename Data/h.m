
function z = h(x)
N = size(x,2);
z = zeros(2,N);
for i = 1:N
    z(1,i) = x(1, i)^2;
    z(2,i) = x(2, i);
end
end

% function z = h(x)
% z = x;
% end


function [y] = decimation(x,M)
  % decimation by integer matrix M
  V = x.gen;
  U = V*M;
  [D,L] = size(x.n);
  x_r = V*x.n;
  
  temp = inv(U)*x_r; % if a column has only integers it's on the new lattice
  % this is actually the new n vectors under the new generator
  temp2 = floor(temp)==temp; %check for integers
%initialize y
  y.n = [];
  y.data = [];
  y.gen = U

  for i = 1:L
    if(sum(temp2(:,i))==D)
    y.n = [y.n,temp(:,i)];
    y.data = [y.data,x.data(i)];
  end
end

end
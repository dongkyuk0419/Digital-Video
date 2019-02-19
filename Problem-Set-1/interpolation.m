function [y] = interpolation(x,M)
  U = x.gen;
  V = U*inv(M);
  
  y.gen = V;
  y.n = M*x.n; % so that y_r == x_r
  y.data = x.data; 
  end
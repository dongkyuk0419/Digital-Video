function [out] = myFT(x,f)
%I'll assume that f is a column vector
[a,~] = size(f);
if a == 1
  f = f.';
end
x_r = x.gen*x.n;
out = x.data*(exp(1i*2*pi*(f.'*x_r)))'; %hermitian operator absorbs the negative.
end
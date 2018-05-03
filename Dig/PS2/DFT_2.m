function [ out ] = DFT_2(in,k1,k2,N1,N2)
[m,n] = size(in);
n1 = repmat(0:m-1,m,1);
n2 = repmat(0:n-1,n,1).';
theta = k1/N1*n1 + k2/N2*n2;
out = in.*exp(-1i*2*pi*theta);
out = sum(sum(out));
end

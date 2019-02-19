function [ block ] = dct_restore( dct )
%converts quantized dct coeffs into a block.
dct_rec = zeros(size(dct));
dct_rec(1) = dct(1)/255;
dct_rec(2:end) = (dct(2:end)-128)/8;
temp = idct2(dct_rec,[8,8]);
% according to the document
% A_mn = a_p*a_q*cos*cos*B_pq
% B_pq is from -16 to 16 technically
% but for me
% B_pq max is 4
% so result A_mn will be -1 to 1
% so I need to make it to be 0 to 1
block = (temp+1)/2;


end


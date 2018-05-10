function [ out ] = dct_quant( block ,K)
% Converts a block into a dct coefficients that are quantized.
% K is the factor of how many AC componnets to keep
dct = zeros(size(block));
out = 128*ones(size(block));
dct_coeff = dct2(block,[8,8]);
% according to the document
% B_pq = a_p * a_q * sum(sum(A_mn * cos * cos))
% a_p and a_q are bounded by sqrt(1/4) hence 0.5
% cosines are bounded from -1 to 1
% A_mn is bounded by 1
% so maximum that the summation can give is 64
% max B_pq = 64/4 = 16;, min B_pq = -16
% However let's say maximum is 4

out(1) = round(dct_coeff(1)*255);
dct(2:end) = round(dct_coeff(2:end)*32+128); %magic number comes from 128/4
[~,origin] = sort(abs(dct_coeff(2:end)), 'descend' );
out(origin(1:K)) = dct(origin(1:K));
end


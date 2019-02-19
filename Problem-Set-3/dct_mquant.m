function [ out ] = dct_mquant( block ,M)
% Converts a block into a dct coefficients that are quantized.
% M is the MQUANT vector
dct_coeff = dct2(block,[8,8]);
out = (round(255*dct_coeff./ M).*M/255);
end
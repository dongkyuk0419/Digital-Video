function [ PSNR ] = myPSNR( I,P )
    [h,w] = size(I);    
    dfd = I-P;    
    R = max(max(abs(dfd)));
    MSE = sum(sum((dfd.^2)))/h/w;
    PSNR = 10*log10(R^2/MSE);


end


function [ MQUANT ] = myMQUANT( frame )
        [h,w] = size(frame);
        dct_store = zeros(63,floor(h/8),floor(w/8));
        for k = 1:floor(h/8)
            for kk = 1:floor(w/8)
                temp = frame(1+8*(k-1):8*k,1+8*(kk-1):8*kk);        
                dct_result = dct2(temp,[8,8]);
                dct_store(:,k,kk) = dct_result(2:end);
            end
        end
        sigma = squeeze(mean(mean(dct_store.^2,2),3));
        sigma_bar = geomean(geomean(sigma));        
        bitalloc = floor(log2(sigma/sigma_bar)/2 - min(log2(sigma/sigma_bar)/2)+1);
        bitalloc_matrix = zeros(8);
        bitalloc_matrix(2:end) = bitalloc(:);
        bitalloc_matrix(1) = 8;
        MQUANT = 2.^(8-bitalloc_matrix);
end
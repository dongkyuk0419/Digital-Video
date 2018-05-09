function [mv,dfd,P_pred,PSNR,counter] = HBMA(I,P,sr,bs,lvl,counter,err)
% Hierarchical Block Motion Algorithm
% Detailed explanation goes here
if (nargin < 6)
    counter = 0;
end
if (nargin < 7)
    err = 1e5;
end

level_n = length(lvl);
[h,w] = size(I);
P_pred = I;
% P_pred = imresize(I,2,'bilinear');

% Downsample for first level and parameters
I1 = imresize(I,1/lvl(1));
P1 = imresize(P,1/lvl(1));
sr1 = sr/lvl(1);
h1 = h/lvl(1);
w1 = w/lvl(1);
mv = zeros(2,floor(h1/bs(1)),floor(w1/bs(2)));
for i = 1:floor(h1/bs(1))
    for ii = 1:floor(w1/bs(2))
        ref_pix = [bs(1)*(i-1)+1,bs(2)*(ii-1)+1];
        [mv(:,i,ii),counter] = HBMA_mod(I1,P1,bs,ref_pix,sr1,h1,w1,[0;0],counter,err);
    end
end
%For subsequence levels
for kkk = 1:level_n-1
%     half_pel = 0;
%     if (lvl(kkk+1) ==1)
%         half_pel = 1;
%     end
    mult = lvl(kkk+1);
    mv_norm = mv*lvl(kkk)/mult;    
    h_temp = h/mult;
    w_temp = w/mult;
    I_temp = imresize(I,1/mult);
    P_temp = imresize(P,1/mult);
    sr_temp = sr/mult;
    mv_temp = zeros(2,floor(h_temp/bs(1)),floor(w_temp/bs(2)));
    
    for i = 1:floor(h_temp/bs(1))
        for ii = 1:floor(w_temp/bs(2))
            m_i = floor(i/lvl(kkk)*mult);
            m_ii = floor(ii/lvl(kkk)*mult);
            %avoid 0 index
            m_i = m_i+(m_i==0);
            m_ii = m_ii+(m_ii==0);
            ref_pix = [bs(1)*(i-1)+1,bs(2)*(ii-1)+1];
            [mv_temp(:,i,ii),counter] = HBMA_mod(I_temp,P_temp,bs,ref_pix,...
                sr_temp,h_temp,w_temp,mv_norm(:,m_i,m_ii),counter,err);
        end
    end
    mv = mv_temp;
end
for i = 1:floor(h/bs(1))
    for ii = 1:floor(w/bs(2))
        ref_pix = [bs(1)*(i-1)+1,bs(2)*(ii-1)+1];
        I_block = I(ref_pix(1):ref_pix(1)+bs(1)-1,ref_pix(2):...
            ref_pix(2)+bs(2)-1);    
        P_pred(ref_pix(1)+mv(1,i,ii):ref_pix(1)+bs(1)-1+mv(1,i,ii),...
            ref_pix(2)+mv(2,i,ii):ref_pix(2)+bs(2)-1+mv(2,i,ii)) = I_block;
    end
end
%     P_pred = imresize(P_pred,1/2);
    P_pred = P_pred(1:h,1:w);
    dfd = P_pred - P;
    R = max(max(abs(dfd)));
    MSE = sum(sum((dfd.^2)))/h/w;
    PSNR = 10*log10(R^2/MSE);
end
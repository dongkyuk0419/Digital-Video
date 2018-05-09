function [mv,counter] = HBMA_mod(I,P,bs,ref_pix,sr,h,w,mv_norm,counter,err)
% HBMA modular component.
% This function does EBMA on each block.
% half_pel = 1 indicates that it's half-pel resolution
I_block = I(ref_pix(1):ref_pix(1)+bs(1)-1,ref_pix(2):...
    ref_pix(2)+bs(2)-1);
mv = [0;0];


if nargin <10
    err = 1e5;
end
for k = -sr(1):sr(1) % search vertically
    for kk = -sr(2):sr(2) %search horizontally
        if(ref_pix(1)+k+mv_norm(1) <= 0)
            continue
        end
        if(ref_pix(1)+bs(1)-1 + k +mv_norm(1)> h)
            continue
        end
        if(ref_pix(2)+kk +mv_norm(2)<= 0)
            continue
        end
        if(ref_pix(2)+bs(2)-1 + kk +mv_norm(2)> w)
            continue
        end
        P_block = P(ref_pix(1)+k+mv_norm(1):ref_pix(1)+bs(1)-1+k+mv_norm(1),...
            ref_pix(2)+kk+mv_norm(2):ref_pix(2)+bs(2)-1+kk+mv_norm(2));
        MAD = sum(sum(abs(P_block - I_block)));
        counter = counter + 2*h*w -1;
        
        if MAD < err
            err = MAD;
            mv = [k;kk];
        end
    end
end
mv = mv + mv_norm;
counter = counter + 2;
end

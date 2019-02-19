function [ mv,dfd,P_pred,PSNR,counter] = EBMA( I, P, sr, bs,counter)
% Exhaustive Block Motion Algorithm
% I: reference frame
% P: target frame
% sr: search range %[a,b] means search -a:a and -b:b
% bs: block size
%
% mv: motion vector
% dfd: dfd of predicted frame with respect to the original target frame
% P_red: predicted target frame
% PSNR: peak signal to noise ratio
    if nargin < 5
        counter = 0;
    end

    [h,w] = size(I);
    mv = zeros(2,floor(h/bs(1)),floor(w/bs(2))); %motion vector store
    P_pred = I;
    for i = 1:size(mv,2) %iterate through vertical blocks
        for ii = 1:size(mv,3) %iterate through horizontal blocks

            % Motion Vector Estimation
            err = 1e5; %place holder %maximum MAD
            mv_store = [0;0];
            %location of upper left corner 
            ref_pix = [bs(1)*(i-1)+1,bs(2)*(ii-1)+1];
            I_block = I(ref_pix(1):ref_pix(1)+bs(1)-1,ref_pix(2):...
                ref_pix(2)+bs(2)-1);
            for k = -sr(1):sr(1) % search vertically
                for kk = -sr(2):sr(2) %search horizontally

                    % check boundaries (up, down, left, right)
                    if(ref_pix(1) + k <= 0)
                        continue
                    end
                    if(bs(1)*(i) + k > h)
                        continue
                    end
                    if(ref_pix(2) + kk <= 0)
                        continue
                    end
                    if(bs(2)*(ii) + kk > w)
                        continue
                    end
                    % Block Matching
                    P_block = P(ref_pix(1)+k:ref_pix(1)+bs(1)-1+k,...
                        ref_pix(2)+kk:ref_pix(2)+bs(2)-1+kk);
                    frame_diff = I_block - P_block;
                    MAD = sum(sum(abs(frame_diff)));
                    counter = counter + 2*h*w -1;
                    
                    % I chose MAD for computational advantage, and I
                    % removed dividing by number of pixels for speed
                    if(MAD < err)
                        err = MAD;
                        mv_store = [k,kk];
                    end
                end
            end
            mv(:,i,ii) = mv_store;

            % Prediction
            P_pred(ref_pix(1)+mv(1,i,ii):ref_pix(1)+bs(1)-1+mv(1,i,ii),...
                ref_pix(2)+mv(2,i,ii):ref_pix(2)+bs(2)-1+mv(2,i,ii)) = I_block;
        end
    end
    dfd = P_pred - I;
    R = max(max(abs(dfd)));
    MSE = sum(sum((dfd.^2)))/h/w;
    PSNR = 10*log10(R^2/MSE);
end
% DongKyu Kim
% Problem Set 3
% ECE 418 Digital Video
% Professor Fontaine
clc; clear all; close all;
%% 0. Prelim
% MATLAB R2016b was used
video_truth = VideoReader('traffic.mj2');
% implay('traffic.mj2',video_truth.FrameRate)

% Extract frames and convert to grayscale
frames = zeros(video_truth.Height, video_truth.Width, video_truth.FrameRate...
    *video_truth.Duration);
for i = 1:size(frames,3)
    frames(:,:,i) = rgb2gray(read(video_truth,i));
end
frames = frames/255;
% frames = uint8(frames);
% implay(frames,video_truth.FrameRate)

%% 1. EBMA
% I will do the EBMA for the 100th and 101th frame.
I_frame_n = 100;
P_frame_n = 101;

sr = [32,32]; % search -32 to 32 for both ways
bs = [16,16]; % block size of 16 x 16
counter = 0;
[mv,dfd,P_pred,PSNR,counter] = EBMA(frames(:,:,I_frame_n),frames(:,:,P_frame_n),sr,bs,counter);
X = ['This frame took ', num2str(counter), ' additions.'];
disp(X);
% Original 100th and 101th frame
figure;
subplot(2,3,1);
imshow(frames(:,:,I_frame_n));
title('I frame');

subplot(2,3,2);
imshow(frames(:,:,P_frame_n));
title('P frame original');

% 100th frame with motion vector
subplot(2,3,3);
[x,y] = meshgrid((1:bs(2):size(mv,3)*bs(2))+bs(2)/2-1,(1:bs(1):size(mv,2)...
    *bs(1))+bs(1)/2-1);
imshow(frames(:,:,I_frame_n));
hold on
quiver(x,y,squeeze(mv(2,:,:)),squeeze(mv(1,:,:)))
hold off
title('I frame with motion vector');

% Predicted 101th frame
subplot(2,3,5);
imshow(P_pred);
title('Predcted P frame');

% DFD
subplot(2,3,6);
imshow(dfd);
title('DFD of predicted P frame from EBMA');

X = ['PSNR of this frame is ', num2str(PSNR), ' dB.'];
disp(X);
% I experimented with how PSNR degrades as you go farther away from the I
% frame. It starts off with about 20dB, but at 4 frames away, the PSNR
% degrades to 16dB, while at 3 frames away, PSNR is still in 17dB.
% So I'm guessing 3 P frames per I frame makes a good encoding parameter.

% % Movie after EBMA
% frames_P = zeros(size(frames));
% PSNR_whole = zeros(1,size(frames,3));
% counter = 0;
% for i = 1:4:size(frames,3)
%     frames_P(:,:,i) = frames(:,:,i);
%     PSNR_whole(i) = 0;
%     [~,~,frames_P(:,:,i+1),PSNR_whole(i+1),counter] = EBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,counter);
%     [~,~,frames_P(:,:,i+2),PSNR_whole(i+2),counter] = EBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,counter);
%     [~,~,frames_P(:,:,i+3),PSNR_whole(i+3),counter] = EBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,counter);
% end
% X = ['This video took ', num2str(counter), ' additions.'];
% disp(X);
% % This video took 717965302500 additions.

disp('Playing the video predicted with EBMA');
load('frames_P.mat'); % I exported the data so that I don't have to compute above twice.
load('PSNR_whole');
implay(frames_P,video_truth.FrameRate)
X = ['Max PSNR of this video is ', num2str(max(PSNR_whole)), ' dB.'];
X2 = ['Min PSNR of this video is ', num2str(min(PSNR_whole(PSNR_whole~=0))),' dB.'];
disp(X);
disp(X2);

%% 2. HBMA
% I will do the EBMA for the 100th and 101th frame.
I_frame_n = 100;
P_frame_n = 101;

bs = [16,16];
sr = [32,32];
% lvl1 = [8 4 2 1]; % my example video is too low resolution
lvl2 = [4 2 1];
counter = 0;
[mv,dfd,P_pred,PSNR,counter] = HBMA(frames(:,:,I_frame_n),frames(:,:,P_frame_n),sr,bs,lvl2,counter);
X = ['This frame took ', num2str(counter), ' additions.'];
disp(X);

% Original 100th and 101th frame
figure;
subplot(2,3,1);
imshow(frames(:,:,I_frame_n));
title('I frame');

subplot(2,3,2);
imshow(frames(:,:,P_frame_n));
title('P frame original');

% 100th frame with motion vector
subplot(2,3,3);
[x,y] = meshgrid((1:bs(2):size(mv,3)*bs(2))+bs(2)/2-1,(1:bs(1):size(mv,2)...
    *bs(1))+bs(1)/2-1);
imshow(frames(:,:,I_frame_n));
hold on
quiver(x,y,squeeze(mv(2,:,:)),squeeze(mv(1,:,:)))
hold off
title('I frame with motion vector');

% Predicted 101th frame
subplot(2,3,5);
imshow(P_pred);
title('Predcted P frame');

% DFD
subplot(2,3,6);
imshow(dfd);
title('DFD of predicted P frame from HBMA');

X = ['PSNR of this frame is ', num2str(PSNR), ' dB.'];
disp(X);

% % Movie after HBMA
% frames_P_H = zeros(size(frames));
% PSNR_whole_H = zeros(1,size(frames,3));
% dfd_H = zeros(size(frames));
% counter = 0;
% for i = 1:4:size(frames,3)
%     frames_P_H(:,:,i) = frames(:,:,i);
%     PSNR_whole_H(i) = 0;
%     [~,dfd_H(:,:,i+1),frames_P_H(:,:,i+1),PSNR_whole_H(i+1),counter] = HBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,lvl2,counter);
%     [~,dfd_H(:,:,i+2),frames_P_H(:,:,i+2),PSNR_whole_H(i+2),counter] = HBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,lvl2,counter);
%     [~,dfd_H(:,:,i+3),frames_P_H(:,:,i+3),PSNR_whole_H(i+3),counter] = HBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,lvl2,counter);
% end
% X = ['This video took ', num2str(counter), ' additions.'];
% disp(X);
% % This video took 728867681770 additions.

disp('Playing the video predicted with HBMA');
load('frames_P_H.mat'); % I exported the data so that I don't have to compute above twice.
load('PSNR_whole_H');
implay(frames_P_H,video_truth.FrameRate)
X = ['Max PSNR of this video is ', num2str(max(PSNR_whole_H)), ' dB.'];
X2 = ['Min PSNR of this video is ', num2str(min(PSNR_whole_H(PSNR_whole_H~=0))),' dB.'];
disp(X);
disp(X2);

%% 3 Modified HBMA
I_frame_n = 100;
P_frame_n = 101;

bs = [16,16];
sr = [32,32];
% lvl1 = [8 4 2 1]; % my example video is too low resolution
lvl2 = [4 2 1];
counter = 0;
err = 30; % this is DFD control in essence, determined iteratively
[mv,dfd,P_pred,PSNR,counter] = HBMA(frames(:,:,I_frame_n),frames(:,:,P_frame_n),sr,bs,lvl2,counter,err);
X = ['This frame took ', num2str(counter), ' additions.'];
disp(X);

% Original 100th and 101th frame
figure;
subplot(2,3,1);
imshow(frames(:,:,I_frame_n));
title('I frame');

subplot(2,3,2);
imshow(frames(:,:,P_frame_n));
title('P frame original');

% 100th frame with motion vector
subplot(2,3,3);
[x,y] = meshgrid((1:bs(2):size(mv,3)*bs(2))+bs(2)/2-1,(1:bs(1):size(mv,2)...
    *bs(1))+bs(1)/2-1);
imshow(frames(:,:,I_frame_n));
hold on
quiver(x,y,squeeze(mv(2,:,:)),squeeze(mv(1,:,:)))
hold off
title('I frame with motion vector');

% Predicted 101th frame
subplot(2,3,5);
imshow(P_pred);
title('Predcted P frame');

% DFD
subplot(2,3,6);
imshow(dfd);
title('DFD of predicted P frame from HBMA with DFD control');

X = ['PSNR of this frame is ', num2str(PSNR), ' dB.'];
disp(X);


% % Movie after HBMA with Threshold
% frames_P_H2 = zeros(size(frames));
% PSNR_whole_H2 = zeros(1,size(frames,3));
% dfd_H2 = zeros(size(frames));
% counter = 0;
% for i = 1:4:size(frames,3)
%     frames_P_H2(:,:,i) = frames(:,:,i);
%     PSNR_whole_H2(i) = 0;
%     [~,dfd_H2(:,:,i+1),frames_P_H2(:,:,i+1),PSNR_whole_H2(i+1),counter] = HBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,lvl2,counter,err);
%     [~,dfd_H2(:,:,i+2),frames_P_H2(:,:,i+2),PSNR_whole_H2(i+2),counter] = HBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,lvl2,counter,err);
%     [~,dfd_H2(:,:,i+3),frames_P_H2(:,:,i+3),PSNR_whole_H2(i+3),counter] = HBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,lvl2,counter,err);
% end
% X = ['This video took ', num2str(counter), ' additions.'];
% disp(X);
% % This video took 729206727044 additions.

disp('Playing the video predicted with HBMA with Threshold');
load('frames_P_H2.mat'); % I exported the data so that I don't have to compute above twice.
load('PSNR_whole_H2');
implay(frames_P_H2,video_truth.FrameRate)
X = ['Max PSNR of this video is ', num2str(max(PSNR_whole_H2)), ' dB.'];
X2 = ['Min PSNR of this video is ', num2str(min(PSNR_whole_H2(PSNR_whole_H2~=0))),' dB.'];
disp(X);
disp(X2);

% DFD graphs
load('dfd_H');
load('dfd_H2');
msq_dfd1 = squeeze(mean(mean((dfd_H.^2),1),2));
msq_dfd2 = squeeze(mean(mean((dfd_H2.^2),1),2));
figure;
temp = linspace(0,0.03,13);
subplot(2,1,1);
h = histogram(msq_dfd1,temp);
title('histogram for HBMA without Threshold');
subplot(2,1,2);
histogram(msq_dfd2,temp);
xlabel('mean squared dfd');
title('histogram for HBMA with Threshold');
% You can sort of see that some higher value dfd has shifted towards lower
% dfds.
%% 4 DCT
K = 41; % how many AC to keep
sample_block = frames(1:8,1:8,1);
sample_dct_quant = dct_quant(sample_block,K);
sample_block_recovered = dct_restore(sample_dct_quant);
% 
% X = ['PSNR is ', num2str(myPSNR(sample_block,sample_block_recovered)), ' dB.'];
% disp(X);
% % 
% figure;
% subplot(2,1,1)
% imshow(sample_block);
% title('original block');
% subplot(2,1,2)
% imshow(sample_block_recovered);
% title('restored block');

sample_frame = frames(:,:,1);
% Trying this on a whole frame
% 
% storage = zeros(1,63); % to find optimal K
% for K = 1:63
    recov_frame = zeros(size(sample_frame));
    for k = 1:size(sample_frame,1)/8
        for kk = 1:size(sample_frame,2)/8
            temp = sample_frame(1+8*(k-1):8*k,1+8*(kk-1):8*kk);
            sample_dct_quant = dct_quant(temp,K);
            recov_frame(1+8*(k-1):8*k,1+8*(kk-1):8*kk) = dct_restore(sample_dct_quant);
        end
    end
%     storage(K) = myPSNR(sample_frame,recov_frame);
%     [~,rankings] = sort(storage, 'descend' );
% end


figure;
subplot(2,1,1)
imshow(sample_frame);
title('original block 153600 bits');
subplot(2,1,2)
imshow(recov_frame);
title('restored block 100800 bits');
X = ['PSNR of this frame is ', num2str(myPSNR(sample_frame,recov_frame)), ' dB.'];
disp(X);
% This kind of works K = 40~45 tends to yield the best range
% and K = 41 did the best. So I think I'm using K = 41.

% This yield 65% reduction in bits.
%% 5 MQUANT
load('dfd_H2');
        sample_dfd_block = dfd_H2(:,:,100);
        [h,w] = size(sample_dfd_block);
        dct_store = zeros(63,floor(h/8),floor(w/8));
        for k = 1:floor(h/8)
            for kk = 1:floor(w/8)
                temp = sample_dfd_block(1+8*(k-1):8*k,1+8*(kk-1):8*kk);        
                dct_result = dct2(temp,[8,8]);
                dct_store(:,k,kk) = dct_result(2:end);
            end
        end
        sigma = squeeze(mean(mean(dct_store.^2,2),3));
        sigma_bar = geomean(geomean(sigma));

        figure;
        stem(log2(sigma));
        xlabel('slot number');
        title('stem plot of log_2(\sigma_{k}^2)');
        
        % stem plot 
        bitalloc = floor(log2(sigma/sigma_bar)/2 - min(log2(sigma/sigma_bar)/2)+1);
        bitalloc_matrix = zeros(8);
        bitalloc_matrix(2:end) = bitalloc(:);
        bitalloc_matrix(1) = 8;
        MQUANT = 2.^(8-bitalloc_matrix);
% I stored this routine without the figure portion to myMQUANT.m
       
        
%% 6a Codec - Encoding

video_truth = VideoReader('traffic.mj2');
% This whole thing could go into a function. But I wanted plots
dfd_enc = zeros(size(frames));
counter = 0; % so that code runs
mv_enc=zeros(2,7,10,120); %hard coded
% HBMA
for i = 1:4:size(frames,3)  
    [mv_enc(:,:,:,i+1),dfd_enc(:,:,i+1),~,~] = HBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,lvl2,counter,err);
    [mv_enc(:,:,:,i+2),dfd_enc(:,:,i+2),~,~] = HBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,lvl2,counter,err);
    [mv_enc(:,:,:,i+3),dfd_enc(:,:,i+3),~,~] = HBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,lvl2,counter,err);
end

% Encoding loop
MQUANT_store = zeros(8,8,size(frames,3));
encoded = zeros(size(frames));
num_bits = zeros(size(frames,3));

for i = 1:size(frames,3)
    if (mod(i,4) == 1)
        encoded(:,:,i) = round(frames(:,:,i)*255/8)*8/255;   % This is doing 8 bits
        num_bits(i) = size(frames,1)*size(frames,2)*8;
        continue
    end
    MQUANT_store(:,:,i) = myMQUANT(dfd_enc(:,:,i));
    
    target = dfd_enc(:,:,i);
    temp2 = zeros(size(target));
    for k = 1:size(target,1)/8
        for kk = 1:size(target,2)/8
            temp = target(1+8*(k-1):8*k,1+8*(kk-1):8*kk);
            temp2(1+8*(k-1):8*k,1+8*(kk-1):8*kk) = dct_mquant(temp,MQUANT_store(:,:,i));
        end
    end
    
    encoded(:,:,i) = temp2;
    num_bits(i) = sum(sum(8-log2(MQUANT_store(:,:,i))))*size(frames,1)*size(frames,2)/8/8;
end

% I didn't know exactly how to deel with motion vectors so I just did
% the  manhattan distance that the motion vector travels
figure;
histogram(sum(abs(mv_enc),1))
title('Histogram of Manhattan Distance that MV travels');
xlabel('Distance');
ylabel('Frequency');

figure;
n = 1:120;
stem(n,num_bits);
title('Number of Bits per Frame');
xlabel('Frame Number');
ylabel('Bits');

%%  6b Codec - Decoding
%Decoding
decoded = zeros(size(frames));
for i = 1: size(frames,3)
    if (mod(i,4) == 1)
        decoded(:,:,i) = encoded(:,:,i);
        continue
    end
    target = encoded(:,:,i);
    temp2 = zeros(size(target));
    for k = 1:size(target,1)/8
        for kk = 1:size(target,2)/8
            temp = target(1+8*(k-1):8*k,1+8*(kk-1):8*kk);
            temp2(1+8*(k-1):8*k,1+8*(kk-1):8*kk) = idct2(temp,[8,8]);
        end
    end
    decoded(:,:,i) = encoded(:,:,i-(mod(i-1,4))) + temp2;
end
disp('playing the decoded video')
    implay(decoded,video_truth.FrameRate)

    
% DongKyu Kim
% Problem Set 3
% ECE 418 Digital Video
% Professor Fontaine
clc; clear all; close all;
%% 0. Prelim
% MATLAB R2016b was used
video_truth = VideoReader('traffic.mj2');
% implay('traffic.mj2')

% Extract frames and convert to grayscale
frames = zeros(video_truth.Height, video_truth.Width, video_truth.FrameRate...
    *video_truth.Duration);
for i = 1:size(frames,3)
    frames(:,:,i) = rgb2gray(read(video_truth,i));
end
frames = frames/255;
% frames = uint8(frames);
% implay(frames)

%% 1. EBMA
% I will do the EBMA for the 100th and 101th frame.
I_frame_n = 100;
P_frame_n = 101;

sr = [32,32]; % search -32 to 32 for both ways
bs = [16,16]; % block size of 16 x 16
[mv,dfd,P_pred,PSNR] = EBMA(frames(:,:,I_frame_n),frames(:,:,P_frame_n),sr,bs);

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

% Movie after EBMA
% frames_P = zeros(size(frames));
% PSNR_whole = zeros(1,size(frames,3));
% for i = 1:4:size(frames,3)
%     frames_P(:,:,i) = frames(:,:,i);
%     PSNR_whole(i) = 0;
%     [~,~,frames_P(:,:,i+1),PSNR_whole(i+1)] = EBMA(frames(:,:,i),frames(:,:,i+1),sr,bs);
%     [~,~,frames_P(:,:,i+2),PSNR_whole(i+2)] = EBMA(frames(:,:,i),frames(:,:,i+2),sr,bs);
%     [~,~,frames_P(:,:,i+3),PSNR_whole(i+3)] = EBMA(frames(:,:,i),frames(:,:,i+3),sr,bs);
% end
disp('Playing the video predicted with EBMA');
load('frames_P.mat'); % I exported the data so that I don't have to compute above twice.
load('PSNR_whole');
implay(frames_P)
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
lvl1 = [8 4 1];
lvl2 = [8 1];
[mv,dfd,P_pred,PSNR] = HBMA(frames(:,:,I_frame_n),frames(:,:,P_frame_n),sr,bs,lvl1)






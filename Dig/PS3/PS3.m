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

% Movie after EBMA
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
frames_P_H = zeros(size(frames));
PSNR_whole_H = zeros(1,size(frames,3));
counter = 0;
for i = 1:4:size(frames,3)
    frames_P_H(:,:,i) = frames(:,:,i);
    PSNR_whole_H(i) = 0;
    [~,~,frames_P_H(:,:,i+1),PSNR_whole_H(i+1),counter] = HBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,lvl2,counter);
    [~,~,frames_P_H(:,:,i+2),PSNR_whole_H(i+2),counter] = HBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,lvl2,counter);
    [~,~,frames_P_H(:,:,i+3),PSNR_whole_H(i+3),counter] = HBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,lvl2,counter);
end
X = ['This video took ', num2str(counter), ' additions.'];
disp(X);
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
err = 3; % this is DFD control in essence
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


% % Movie after HBMA
frames_P_H = zeros(size(frames));
PSNR_whole_H = zeros(1,size(frames,3));
counter = 0;
for i = 1:4:size(frames,3)
    frames_P_H(:,:,i) = frames(:,:,i);
    PSNR_whole_H(i) = 0;
    [~,~,frames_P_H(:,:,i+1),PSNR_whole_H(i+1),counter] = HBMA(frames(:,:,i),frames(:,:,i+1),sr,bs,lvl2,counter,err);
    [~,~,frames_P_H(:,:,i+2),PSNR_whole_H(i+2),counter] = HBMA(frames(:,:,i),frames(:,:,i+2),sr,bs,lvl2,counter,err);
    [~,~,frames_P_H(:,:,i+3),PSNR_whole_H(i+3),counter] = HBMA(frames(:,:,i),frames(:,:,i+3),sr,bs,lvl2,counter,err);
end
X = ['This video took ', num2str(counter), ' additions.'];
disp(X);
% % This video took 728867681770 additions.

% disp('Playing the video predicted with HBMA');
% load('frames_P_H.mat'); % I exported the data so that I don't have to compute above twice.
% load('PSNR_whole_H');
% implay(frames_P_H,video_truth.FrameRate)
% X = ['Max PSNR of this video is ', num2str(max(PSNR_whole_H)), ' dB.'];
% X2 = ['Min PSNR of this video is ', num2str(min(PSNR_whole_H(PSNR_whole_H~=0))),' dB.'];
% disp(X);
% disp(X2);


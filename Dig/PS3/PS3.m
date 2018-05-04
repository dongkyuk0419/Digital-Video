% DongKyu Kim
% Problem Set 3
% ECE 418 Digital Video
% Professor Fontaine
clc; clear all; close all;
%% 0. Prelim
video_truth = VideoReader('traffic.mj2');
% implay('traffic.mj2')
frames = zeros(video_truth.Height, video_truth.Width, video_truth.FrameRate*video_truth.Duration);
% 
while
for i = 1:size(frames,3)
    frames(:,:,i) = uint8(rgb2gray(readFrame(video_truth,i)));
end
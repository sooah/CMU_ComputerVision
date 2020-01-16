clc; clear all; close all;

% f = zeros(100,100);
% 
% for i = 1:size(f,1)
%     f(i,i) = 1;
%     figure(10); imshow(f, []); hold on;
%     plot(f(i,i),'g'); hold on;
% end

% [h,t,r] = hough([10,10]);
f = zeros(100,100);
f(10,10) = 1;
% H = hough(f);
% imshow(H,[]);

f(20,20) = 1;
% H = hough(f);
% imshow(H,[]);


f(30,30) = 1;
H = hough(f);
imshow(H,[]);   
hold on;

a = houghpeaks(H);
plot(a(2), a(1), '*r');

x = 0:1:100;
y = tan(-90+a(2))*x;
figure(); plot(x,y);

clear; close all; clc; 

datadir = '../ec/street.jpg';  
addpath('../matlab');
%parameters
sigma     = 2;
threshold = 0.3;
rhoRes    = 2;
thetaRes  = pi/90;
nLines    = 50;
%end of parameters

img = imread(datadir);

if (ndims(img) == 3)
    img = rgb2gray(img);
end

img = double(img) / 255;

%actual Hough line code function calls%  
[Im] = myEdgeFilter(img, sigma);   
[H,rhoScale,thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes);
[rhos, thetas] = myHoughLines(H, nLines);
lines = houghlines(Im>threshold, 180*(thetaScale/pi), rhoScale, [rhos, thetas],'FillGap',5,'MinLength',10);

new_result_path = sprintf('../para');
mkdir(new_result_path);

fname = sprintf('%s/01edge.png', new_result_path);
imwrite(sqrt(Im/max(Im(:))), fname);
fname = sprintf('%s/02threshold.png', new_result_path);
imwrite(Im > threshold, fname);
fname = sprintf('%s/03hough.png', new_result_path);
imwrite(H/max(H(:)), fname);
fname = sprintf('%s/04lines.png', new_result_path);

img2 = img;
for j=1:numel(lines)
   img2 = drawLine(img2, lines(j).point1, lines(j).point2); 
end     
imwrite(img2, fname);



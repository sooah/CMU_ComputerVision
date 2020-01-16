function [img1] = myEdgeFilter(img0, sigma)
%Your implemention

% 1. smooth image
h = 2*ceil(3*sigma)+1;
gaus = fspecial('gaussian', h, sigma);
im = myImageFilter(img0,gaus);


sobel = fspecial('sobel');
imgy = myImageFilter(im, sobel);
imgx = myImageFilter(im, sobel');
% grad = [-imgx;imgy];

% G_dir = abs(atand(imgy./(-imgx)));
G_dir = atand(imgy./(-imgx));
G_dir(G_dir<0) = G_dir(G_dir<0)+180;
G_dir(G_dir>=0 & 22.5>=G_dir) = 0;
G_dir(G_dir>22.5 & 67.5 >= G_dir) = 45;
G_dir(G_dir>67.5 & 112.5 >= G_dir) = 90;
G_dir(G_dir>112.5 & 157.5 >= G_dir) = 135;
G_mag = sqrt(imgx.^2+imgy.^2);

new_img = zeros(size(img0,1), size(img0,2));

for i =1:size(im,1)
    for j = 1:size(im,2)
        mag_current = G_mag(i,j);
        dir_current = G_dir(i,j);
        if dir_current == 0
            try mag_n1 = G_mag(i,j-1); catch mag_n1 = 0; end
            try mag_n2 = G_mag(i+1,j); catch mag_n2 = 0; end
        elseif dir_current == 45
            try mag_n1 = G_mag(i-1,j+1); catch mag_n1 = 0; end
            try mag_n2 = G_mag(i+1,j-1); catch mag_n2 = 0; end
        elseif dir_current == 90
            try mag_n1 = G_mag(i-1,j); catch mag_n1 = 0; end
            try mag_n2 = G_mag(i+1,j); catch mag_n2 = 0; end
        else
            try mag_n1 = G_mag(i-1,j-1); catch mag_n1 = 0; end
            try mag_n2 = G_mag(i+1,j+1); catch mag_n2 = 0; end
        end
        
        if max([mag_current,mag_n1, mag_n2]) ~= mag_current
            new_img(i,j) = 0;
        else
            new_img(i,j) = G_mag(i,j);
        end        
    end
end
img1 = new_img;
end
   
        

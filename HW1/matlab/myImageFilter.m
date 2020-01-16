function [img1] = myImageFilter(img0, h)
% reference of this function is conv2

pad = (size(h,1)-1)/2;
filt = size(h,1);

pad_img = padarray(img0, [pad pad], 'replicate', 'both'); 

for i = 1:size(img0,1) % x size 
    for j = 1:size(img0,2) % y size
        patch = (pad_img(i:i+filt-1,j:j+filt-1));
        conv_patch = h.*patch;
        conv(i,j) = -sum(conv_patch,'all');
    end
end
img1 = conv;
end

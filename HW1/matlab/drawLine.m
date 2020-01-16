function [ img ] = drawLine( img, start, stop )

delta = 0.05 / norm(stop-start);

t = 0:delta:1;

x = start(1) + t * ( stop(1) - start(1) );
y = start(2) + t * ( stop(2) - start(2) );

if(ndims(img) == 2)
    img = repmat(img,[1 1 3]);
end

if (ischar(img))
    img = double(img) / 255;
end

x = round(x);
y = round(y);

idx = y + (x-1) * size(img,1);
img(idx) = 0;
img(idx +   size(img,1) * size(img,2)) = 1;
img(idx + 2*size(img,1) * size(img,2)) = 0;

end


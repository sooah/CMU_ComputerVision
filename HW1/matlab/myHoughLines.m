function [rhos, thetas] = myHoughLines(H, nLines)
%Your implemention here
new_img = zeros(size(H));

for i = 1:size(H,1)
    for j = 1:size(H,2)
        mag_cent = H(i,j);
        try mag(1) = H(i-1,j-1); catch mag(1) = 0; end
        try mag(2) = H(i-1,j); catch mag(2) = 0; end
        try mag(3) = H(i-1,j+1); catch mag(3) = 0; end
        try mag(4) = H(i,j-1); catch mag(4) = 0; end
        try mag(5) = H(i,j+1); catch mag(5) = 0; end
        try mag(6) = H(i+1,j-1); catch mag(6) = 0; end
        try mag(7) = H(i+1,j); catch mag(7) = 0; end
        try mag(8) = H(i+1,j+1); catch mag(8) = 0; end
        mag(9) = mag_cent;
        if max(mag) == mag_cent
            new_img(i,j) = H(i,j);
        else
            new_img(i,j) = 0;
        end
    end
end
flat_val = reshape(new_img,[size(new_img,1)*size(new_img,2),1]);
sort_val = sort(flat_val,'descend');
y = [];
x = [];
for i = 1:nLines
%     try
%         [y(end+1),x(end+1)] = find(new_img==sort_val(i));
%     catch
%         [y(end+1),x(end+1)] = find(new_img==sort_val(i),'first');
%         [y(end+1),x(end+1)] = find(new_img==sort_val(i),'last');
%     end 
% end
    [y_,x_] = find(new_img==sort_val(i));
    y = [y y_'];
    x = [x x_'];
end

rhos = y(1,1:nLines)';
thetas = x(1,1:nLines)';
end
        
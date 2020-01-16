function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
%Your implementation here
Im(Im < threshold) = 0;
[y,x] = find(Im);

theta = (0:thetaRes:2*pi-thetaRes);

for i = 1:size(x,1)
    rho(i,:) = x(i)*cos(theta)+y(i)*sin(theta);
end
rho = round(rho);


% negative invalid
rho(rho<0) = 0;

rho_max = sqrt(size(Im,1).^2+size(Im,2).^2);
h = zeros(ceil(rho_max/rhoRes),size(theta,2));

for i = 1:size(rho,2)
    rho_ = rho(:,i);
    rho_ = round(rho_/rhoRes);
    for j = 1:size(rho_,1)
        idx = rho_(j,1);
        if idx ~= 0
            h(idx,i) = h(idx,i)+1;
        end
    end
end
H = h;
thetaScale = theta;
rhoScale = (rhoRes:rhoRes:rho_max);

end
        
        
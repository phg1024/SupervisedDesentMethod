function stage = computeDescentDirection(trainset, features, target_features)

N = numel(trainset);
Lfp = length(trainset{1}.guess); Nfp = Lfp/2;
Lfeat = Nfp * 128;
dx = zeros(N, Lfp);
dphi = zeros(N, Lfeat);

numel(features)
numel(target_features)

for i=1:N
    dx(i,:) = trainset{i}.truth - trainset{i}.guess;
    dphi(i,:) = cell2mat(features{i}) - cell2mat(target_features{i});
end

cutoff = 98.0;
[coeff,score,latent,tsquared,explained,mu] = pca(dphi);
sumEnergy = 0; bestIdx = 1;
for i=1:length(explained)
    sumEnergy = sumEnergy + explained(i);
    if sumEnergy > cutoff
        bestIdx = i;
    end
end

% projection
dphi_proj = dphi * coeff(:,1:bestIdx);
R = (dphi_proj'*dphi_proj)\(dphi_proj'*dx);
R = R';
b = mean(dx) - (R * mean(dphi_proj)')';
%b = dx - (R * dphi_proj')';

stage.R = R;
stage.pVecs = coeff(:, 1:bestIdx);
stage.b = b;
end
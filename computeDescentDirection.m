function stage = computeDescentDirection(trainset, features, target_features)
fprintf('computing descent direction ...\n');
N = numel(trainset);
Lfp = length(trainset{1}.guess); Nfp = Lfp/2;
Lfeat = length(cell2mat(features{1}));
dx = zeros(N, Lfp);
dphi = zeros(N, Lfeat);

for i=1:N
    dx(i,:) = trainset{i}.truth - trainset{i}.guess;
    dphi(i,:) = cell2mat(features{i}) - cell2mat(target_features{i});
end
fprintf('dx and dphi computed.\n');

tpca = tic;
fprintf('performing pca ...\n');
cutoff = 98.0;
[coeff,score,latent,tsquared,explained,mu] = pca(dphi);
sumEnergy = 0; bestIdx = 1;
for i=1:length(explained)
    sumEnergy = sumEnergy + explained(i);
    if sumEnergy > cutoff
        bestIdx = i;
        break;
    end
end
timecost_pca = toc(tpca);
fprintf('pca finished in %.2f seconds. dimensions = %d\n', timecost_pca, bestIdx);

% projection
dphi_proj = dphi * coeff(:,1:bestIdx);
R = (dphi_proj'*dphi_proj)\(dphi_proj'*dx);
R = R';
b = mean(dx) - (R * mean(dphi_proj)')';

stage.R = R;
stage.pVecs = coeff(:, 1:bestIdx);
stage.b = b;
end
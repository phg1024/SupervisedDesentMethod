function stage = computeDescentDirection(trainset, features, target_features)
fprintf('computing descent direction ...\n');
N = numel(trainset);
Lfp = length(trainset{1}.guess); Nfp = Lfp/2;
dx = zeros(N, Lfp);

phi = features;

for i=1:N
    dx(i,:) = trainset{i}.truth - trainset{i}.guess;
    %dphi(i,:) = features(i,:) - (target_features{i});
end
fprintf('dx and dphi computed.\n');

tpca = tic;
fprintf('performing pca ...\n');
cutoff = 98.0;
if size(phi, 1) > (size(phi,2) / 2)
    [coeff, explained] = fpca(phi);
else
    [coeff, ~, ~, ~, explained] = pca(phi);
end
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
dphi_proj = phi * coeff(:,1:bestIdx);
R = (dphi_proj'*dphi_proj)\(dphi_proj'*dx);
R = R';
b = mean(dx) - (R * mean((features - target_features) * coeff(:,1:bestIdx))')';

stage.R = R;
stage.pVecs = coeff(:, 1:bestIdx);
stage.b = b;
end
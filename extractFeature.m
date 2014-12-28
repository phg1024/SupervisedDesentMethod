function fvectors = extractFeature(training_sample, coords, wsize, nbins, cell_size, nblocks)
Lfp = length(training_sample.guess); Nfp = Lfp / 2;
Lfeat = nbins * nblocks * nblocks;
fvectors = zeros(1, Nfp*Lfeat);
addpath .\features\;
% extract patches
for i=1:Nfp
    x = coords(i, 1); y = coords(i, 2);
    p = extractPatch(training_sample.image, x, y, wsize);
    fvectors((i-1)*Lfeat+1:i*Lfeat) = HoG(p, [nbins, cell_size, nblocks, 1, 0.2])';
end

end


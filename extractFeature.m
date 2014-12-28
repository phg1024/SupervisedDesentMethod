function fvectors = extractFeature(training_sample, coords, wsize, nbins, cell_size, nblocks)
Lfp = length(training_sample.guess); Nfp = Lfp / 2;
fvectors = cell(1, Nfp);
addpath .\features\;
% extract patches
for i=1:Nfp
    x = coords(i, 1); y = coords(i, 2);
    p = extractPatch(training_sample.image, x, y, wsize);
    fvectors{i} = HoG(p, [nbins, cell_size, nblocks, 1, 0.2])';
end

end


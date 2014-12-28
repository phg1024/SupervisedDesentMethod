function fvec = computeFeatureVector(img, x, y, wsize, nbins, cellsize, nblocks)
addpath features\;
p = extractPatch(img, x, y, wsize);
fvec = HoG(p, [nbins, cellsize, nblocks, 1, 0.2])';
end
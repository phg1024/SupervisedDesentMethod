function fvec = computeFeatureVector(img, coords, wsize, nbins, cellsize, nblocks)
addpath features\;
Nfp = size(coords, 1);
Lfeat = nbins*nblocks*nblocks;
fvec = zeros(1, Nfp*Lfeat);
for i=1:Nfp
    p = extractPatch(img, coords(i,1), coords(i,2), wsize);
    fvec((i-1)*Lfeat+1:i*Lfeat) = HoG(p, [nbins, cellsize, nblocks, 1, 0.2])';
end
end
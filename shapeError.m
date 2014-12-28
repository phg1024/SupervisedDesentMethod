function [maxE, meanE] = shapeError(shape, refshape)
Lfp = length(shape); Nfp = Lfp/2;
shape = reshape(shape, Nfp, 2); refshape = reshape(refshape, Nfp, 2);
ds = shape - refshape;
ref_dist = norm(mean(refshape([38 39 41 42],:)) - mean(refshape([44 45 47 48],:)));
E = sqrt(sum(ds.^2, 2)) / ref_dist;
maxE = max(E); meanE = mean(E);
end
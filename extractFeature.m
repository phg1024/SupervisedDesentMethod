function fvectors = extractFeature(training_sample, coords, wsize)
Lfp = length(training_sample.guess); Nfp = Lfp / 2;
fvectors = cell(1, Nfp);
for i=1:Nfp
    x = coords(i, 1); y = coords(i, 2);
    fvectors{i} = computeFeatureVector(training_sample.image, x, y, wsize);    
end

end
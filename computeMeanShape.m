function [meanshape, scaleMean, scaleVar, translationMean, translationVar] = computeMeanShape(dataset)
refshape = dataset{1}.truth;
npts = length(refshape)/2;
% align all other shapes to this shape
nshapes = numel(dataset);
alignedShapes = zeros(nshapes, npts*2);
for i=1:nshapes
    alignedShapes(i, :) = dataset{i}.truth;
end

% mean shape
meanshape = mean(alignedShapes);

% compute the translational and rotational variance by centering the mean
% shape at the center of normalized square
s = zeros(nshapes, 1);
t = zeros(nshapes, 2);

for i=1:nshapes
    centered_meanshape = alignShapeToBox(meanshape, dataset{i}.box);
    [alignedShapes(i,:), ~, ti, si] = alignShape(centered_meanshape, dataset{i}.truth);
    s(i,:) = si;
    t(i,:) = ti' / dataset{i}.box(3);   % relative translation
end
 
% compute the scale variance and translational variance
scaleVar = std(s);
scaleMean = mean(s);
translationVar = std(t);
translationMean = mean(t);
end
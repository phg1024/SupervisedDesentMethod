% align s1 to s0
function [s1, R, t, s] = alignShape(s1, s0)
npts = length(s1)/2;
s1 = reshape(s1, npts, 2);
s0 = reshape(s0, npts, 2);
[s, R, t] = estimateTransform(s1, s0);
s1 = s * R * s1' + repmat(t, 1, npts);
s1 = s1';
s1 = reshape(s1, 1, npts*2);
end
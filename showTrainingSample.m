function showTrainingSample(sample)
img = sample.image; box = sample.box; 
npts = length(sample.truth)/2;
pts = reshape(sample.truth, npts, 2); guess = reshape(sample.guess, npts, 2);
imshow(img);
rectangle('Position',  box, 'EdgeColor', 'r');
hold on;
plot(pts(:,1), pts(:,2), 'og');
plot(guess(:,1), guess(:,2), 'xr');
end
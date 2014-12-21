function shape = alignShapeToBox(shape0, box0, box)
npts = length(shape0)/2;
shape = reshape(shape0, npts, 2);

scale = box(3) / box0(3);

% align point 1 to origin
%x0 = box0(1); y0 = box0(2);
%dx0 = shape(1, 1) - x0; dy0 = shape(1, 2) - y0;
%x = box(1); y = box(2);

% shape = shape - repmat(shape(1,:), npts, 1);
% shape = shape .* scale;
% shape = shape + repmat([x+dx0*scale, y+dy0*scale], npts, 1);

% align the center of the shape to the center of the box
x = box(1); y = box(2);
xc = x + 0.5 * box(3); yc = y + 0.5 * box(3);
shape = shape - repmat(mean(shape), npts, 1);
shape = shape .* scale;
shape = shape + repmat([xc, yc], npts, 1);

shape = reshape(shape, 1, npts*2);
end
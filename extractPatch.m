function p = extractPatch(img, x, y, wsize)
% extract a patch from the given image
% also make sure the patch lies inside the image
% otherwise it is truncated back to the image

% get the patch
hwsize = round(wsize / 2.0);
tlx = x - hwsize; tly = y - hwsize; brx = x + hwsize; bry = y + hwsize;
[h, w, ~] = size(img);

% see if we need to truncate the patch
if tlx < 1
    tlx = 1; brx = tlx + wsize - 1;
end

if tly < 1
    tly = 1; bry = tly + wsize - 1;
end

if brx > w
    brx = w; tlx = brx - wsize + 1;
end

if bry > h
    bry = h; tly = bry - wsize + 1;
end

p = double(img(round(tly:bry), round(tlx:brx)));

end
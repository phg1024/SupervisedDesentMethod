function fvec = computeFeatureVector(img, x, y, wsize)
% get the patch
hwsize = wsize / 2.0;
tlx = x - hwsize; tly = y - hwsize; brx = x + hwsize; bry = y + hwsize;
[h, w, ~] = size(img);
lx = max(tlx, 1); rx = min(brx, w); ty = max(tly, 1); by = min(bry, h);
I = img(round(ty:by), round(lx:rx));

% see if we need to pad the image
if tlx < 1 || tly < 1
    % pad top/left
    I = padarray(I, max(ceil([1-tly, 1-tlx]), 0), 'symmetric', 'pre');
end

if brx > w || bry > h
    I = padarray(I, max(ceil([bry - h, brx - w]), 0), 'symmetric', 'post');
end

% compute gradient
[Gmag, Gdir] = imgradient(I);

% assemble feature vector
% 8 bins per block, 16 blocks in total
cutoffs = 180:-45:-180;
bsize = wsize / 4;
fvec = zeros(1, 128);
for i=1:4
    bybegin = (i-1)*bsize+1;
    byend = bybegin+bsize-1;
    for j=1:4
        bxbegin = (j-1)*bsize+1;
        bxend = bxbegin+bsize-1;
        block = Gdir(bybegin:byend, bxbegin:bxend);
        Gblock = Gmag(bybegin:byend, bxbegin:bxend);
        bidx = ((i-1)*4 + j-1)*8;
        for binIdx = 1:8
            lset = find(block<=cutoffs(binIdx));
            rset = find(block>cutoffs(binIdx+1));
            %fvec(bidx + binIdx) = sum(Gblock(intersect(lset, rset)));
            fvec(bidx + binIdx) = length(intersect(lset, rset));
        end
    end
end

if norm(fvec) < 1e-9
    % nothing to do
else
    fvec = fvec / norm(fvec);
    fvec = min(fvec, 0.2);
    fvec = fvec / norm(fvec);
end
end
function [boxes, points, succeeded] = applyModel_batch(img, model)

%% Load a face detector and an image
cascade_filepath = 'C:\Users\Peihong\Desktop\Code\Libraries\opencv\sources\data\haarcascades';
detector = cv.CascadeClassifier([cascade_filepath, '\', 'haarcascade_frontalface_alt.xml']);

% Preprocess
[h, w, channels] = size(img);
if channels > 1
    img = rgb2gray(img);
end
gr = cv.equalizeHist(img);
I0 = img;

% Detect bounding box
tdetect = tic;
boxes = detector.detect(gr, 'ScaleFactor',  1.3, ...
    'MinNeighbors', 2, ...
    'MinSize',      [30, 30]);
fprintf('face detection time = %.3f seconds.\n', toc(tdetect));

if isempty(boxes)
    boxes = [];
    points = [];
    succeeded = false;
    return;
end

points = cell(numel(boxes), 1);

% Scale it properly
for bidx = 1:numel(boxes)
    boxSize = boxes{1}(3);
    
    % scale the image
    sfactor = model.window_size / boxSize;
    if sfactor == 0.0
        sfactor = 1.0;
    end
    sfactor = 1.0;
    
    I = imresize(I0, sfactor);    
    box = (boxes{bidx}) * sfactor;
    [nh, nw] = size(I);
    
    % cut out a smaller region
    bsize = box(3);
    bcenter = [box(1) + 0.5 * bsize, box(2) + 0.5 * bsize];
    
    % enlarge this region
    cutsize = 4.0 * bsize / 2;
    nbx_tl = max(1, round(bcenter(1) - cutsize)); nby_tl = max(1, round(bcenter(2) - cutsize));
    nbx_br = min(nw, round(bcenter(1) + cutsize)); nby_br = min(nh, round(bcenter(2) + cutsize));

    % cut out image
    newimg = I(nby_tl:nby_br, nbx_tl:nbx_br);
    % get the new bounding box
    newbox = box;
    newbox(1) = newbox(1) - nbx_tl; newbox(2) = newbox(2) - nby_tl;
    ntrials = 5; nneighbors = 3;
    Lfp = 136; Nfp = Lfp/2;
    
    guess = zeros(ntrials, Lfp);
    for j=1:ntrials
        guess(j,:) = alignShapeToBox(model.meanshape, box);
        guess(j,:) = perturbShape(guess(j,:), box, model.scaleMean, model.scaleVar, model.translationMean, model.translationVar);
    end
   
    T = numel(model.stages);
    Lfeat = model.feat_nbins * model.feat_nblocks * model.feat_nblocks;
    features = zeros(ntrials, Nfp*Lfeat);
    for t=1:T
        for j=1:ntrials
            coords = reshape(guess(j,:), Nfp, 2);
            features(j,:) = computeFeatureVector(newimg, coords, ...
                    model.feat_window_size, model.feat_nbins, model.feat_cellsize, model.feat_nblocks);        
            guess(j,:) = guess(j,:) + (features(j,:) * model.stages{t}.pVecs) * model.stages{t}.R' + model.stages{t}.b;
        end
    end    
    if ntrials > 1
        results = guess;
        % pick k best results
        nearestNeighbors = knnsearch(results, mean(results), 'K', nneighbors);
        points{bidx} = mean(results(nearestNeighbors, :));    
    else
        points{bidx} = guess;
    end
    
    
    % restore the correct positions
    points{bidx} = reshape(points{bidx}, Nfp, 2);
    points{bidx} = points{bidx} + repmat([nbx_tl, nby_tl], Nfp, 1);    
    points{bidx} = points{bidx} / sfactor;
end
succeeded = true;
end
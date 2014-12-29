function model = trainModel(opts)

%% Load a face detector and an image
cascade_filepath = 'C:\Users\Peihong\Desktop\Code\Libraries\opencv\sources\data\haarcascades';
detector = cv.CascadeClassifier([cascade_filepath, '\', 'haarcascade_frontalface_alt.xml']);

%% Load all valid input images
if isfield(opts, 'inputfiles')
    ninputs = size(opts.inputfiles, 1);
else
    ninputs = opts.nimgs;
end

trainset_size = 0;
init_trainset = cell(ninputs, 1);
init_shapes = cell(ninputs, 1);
init_boxes = cell(ninputs, 1);
tic;
for t=1:ninputs
    if mod(t, 100) == 0
        fprintf('processed %d images ...\n', t);
    end
    
    if isfield(opts, 'inputfiles')
        imgfile = opts.inputfiles{t, 1};
        ptsfile = opts.inputfiles{t, 2};
    else
        imgfile = [opts.path, opts.prefix, num2index(t, opts.digits), opts.imgext];
        ptsfile = [opts.path, opts.prefix, num2index(t, opts.digits), opts.ptsext];
    end
    
    if ~exist(imgfile, 'file') || ~exist(ptsfile, 'file')
        continue;
    end
    
    boxfile = regexprep(imgfile, '\.\w+$', '.box');
    
    try
        im = imread(imgfile);
        [h, w, channels] = size(im);
        if channels > 1
            im = rgb2gray(im);
        end
        I0 = im;
        gr = cv.equalizeHist(im);
        
        % Detect face
        if exist(boxfile, 'file')
            boxes = loadBoxes(boxfile);
        else
            boxes = detector.detect(gr, 'ScaleFactor',  1.3, ...
                'MinNeighbors', 2, ...
                'MinSize',      [30, 30]);   
            boxes = reshape(boxes, numel(boxes), 1);
            boxes = cell2mat(boxes);
            saveBoxes(boxes, boxfile);
        end

        % Load the annotations and find the correct box
        points = loadPoints(ptsfile, opts.npts);
        
        % Find the valid box
        box = findValidBox(boxes, points);
        
        % Scale the image for faster process        
        scalingFactor = 1.0;
        maxAllowedSize = 3200;
        if max(h, w) > maxAllowedSize
            scalingFactor = maxAllowedSize / max(h, w);
            I0 = imresize(I0, scalingFactor);
        end
        
        points = points * scalingFactor;
        box = box * scalingFactor;
        
        if length(box) == 4
            % Draw results
            if 0
                clf; showImageWithPoints(gr, box, points); pause;
            end
            
            % Scale it again, make the box
            boxSize = box(3);
            sfactor = opts.params.window_size / boxSize;
            if sfactor == 0.0
                sfactor = 1.0;
            end
            
            I0 = imresize(I0, sfactor);
            points = points * sfactor;
            box = (box) * sfactor;
            
            trainset_size = trainset_size + 1;
            npts = size(points, 1);
            init_trainset{trainset_size} = struct('image', I0, 'box', box, 'truth', reshape(points, 1, npts*2));
            init_shapes{trainset_size} = reshape(points, 1, npts*2);
            init_boxes{trainset_size} = box;
        else
            % not a valid training image
        end
    catch err
        disp(imgfile);
        disp(ptsfile);
        rethrow(err);
    end
end
toc;

init_trainset = init_trainset(~cellfun('isempty',init_trainset));
init_shapes = init_shapes(~cellfun('isempty',init_shapes));
fprintf('%d valid training images found.\n', trainset_size);

% Train the model
model = struct();

[meanshape, scaleMean, scaleVar, translationMean, translationVar] = computeMeanShape(init_trainset);

%% Augment the training set first
oversamples = opts.params.oversample_rate;
trainset = cell(trainset_size*oversamples, 1);

idx = 0;
maxError = 0;
totalError = 0;
for t=1:trainset_size
    % create training samples with the mean shape
    for j=1:oversamples
        idx = idx + 1;
        trainset{idx} = init_trainset{t};
                
        % align the guess shape with the ground truth
        %trainset{idx}.guess = alignShape(meanshape, trainset{idx}.truth);   
        trainset{idx}.guess = alignShapeToBox(meanshape, trainset{idx}.box);
        trainset{idx}.guess = perturbShape(trainset{idx}.guess, trainset{idx}.box, scaleMean, scaleVar, translationMean, translationVar);
        
        [maxE, meanE] = shapeError(trainset{idx}.guess, trainset{idx}.truth);
        maxError = max(maxE, maxError);
        totalError = totalError + meanE;
        if 0
            clf;showTrainingSample(trainset{idx});pause;
        end
    end
end
fprintf('training data augmentation finished. %d training samples in total.\n', numel(trainset));
fprintf('max error = %.6f, mean error = %.6f\n', maxError, totalError / numel(trainset));

%% begin the training process
Nfp = opts.npts; Lfp = Nfp * 2;
stages = cell(opts.params.nstages, 1);
Nsamples = numel(trainset);
Lfeat = opts.params.nbins * opts.params.nblocks * opts.params.nblocks;

% extract feature vectors for the target shapes
fprintf('extracting target feature vectors ...\n');
target_features = zeros(Nsamples, Lfeat*Nfp);
tic;
parfor t=1:Nsamples
    coords = reshape(trainset{t}.truth, Nfp, 2);    
    target_features(t,:) = extractFeature(trainset{t}, coords, ...
        opts.params.feat_window_size, opts.params.nbins, opts.params.cell_size, opts.params.nblocks);
end
fprintf('done.\n');
toc;

features = zeros(Nsamples, Lfeat * Nfp);
error = zeros(Nsamples, 2);
for i=1:opts.params.nstages
    % for each stage, estimate R and b
    fprintf('stage %d ...\n', i);
    % find out the feature vectors
    fprintf('extracing feature vectors ...\n');
    tic;
    parfor t=1:Nsamples
        coords = reshape(trainset{t}.guess, Nfp, 2);   
        features(t,:) = extractFeature(trainset{t}, coords, ...
            opts.params.feat_window_size, opts.params.nbins, opts.params.cell_size, opts.params.nblocks);
    end
    fprintf('done.\n');
    toc;
    
    % train the model
    stages{i} = computeDescentDirection(trainset, features, target_features);
    
    % update the guess
    if 0
        for t=1:Nsamples
            trainset{t}.guess = trainset{t}.guess + (stages{i}.R * (cell2mat(features{t}) * stages{i}.pVecs)')' + stages{i}.b;
            error(t,:) = shapeError(trainset{t}.guess, trainset{t}.truth);
            if 0
                clf;showTrainingSample(trainset{t});pause;
            end        
        end
    else
        deltaShape = (features * stages{i}.pVecs) * stages{i}.R';
        for t=1:Nsamples
            trainset{t}.guess = trainset{t}.guess + deltaShape(t,:) + stages{i}.b;
            error(t,:) = shapeError(trainset{t}.guess, trainset{t}.truth);
            if 0
                clf;showTrainingSample(trainset{t});pause;
            end             
        end        
    end
    
    fprintf('max error = %.6f, mean error = %.6f\n', max(error(:,1)), mean(error(:, 2)));
end

model.meanshape = meanshape;
model.scaleVar = scaleVar;
model.scaleMean = scaleMean;
model.translationVar = translationVar;
model.translationMean = translationMean;
model.init_shapes = init_shapes;
model.init_boxes = init_boxes;
model.stages = stages;
model.window_size = opts.params.window_size;
model.feat_window_size = opts.params.feat_window_size;
model.feat_nbins = opts.params.nbins;
model.feat_nblocks = opts.params.nblocks;
model.feat_cellsize = opts.params.cell_size;
end

function idxstr = num2index(v, digits)
idxstr = num2str(v);
while length(idxstr)<digits
    idxstr = ['0', idxstr];
end
end

function points = loadPoints(filename, npts)
fid = fopen(filename, 'r');
textscan(fid, '%s', 3, 'Delimiter', '\n');
points = textscan(fid, '%f %f', npts, 'Delimiter', '\n');
points = cell2mat(points);
fclose(fid);
end

function boxes = loadBoxes(filename)
fid = fopen(filename, 'r');
nboxes = fscanf(fid, '%d', 1);
boxes = zeros(nboxes, 4);
for i=1:nboxes
boxes(i,:) = fscanf(fid, '%d', [1, 4]);
end
fclose(fid);
end

function saveBoxes(boxes, filename)
[nboxes, ~] = size(boxes);
fid = fopen(filename, 'w');
fprintf(fid, '%d\n', nboxes);
for i=1:nboxes
    fprintf(fid, '%d %d %d %d\n', boxes(i,1), boxes(i,2), boxes(i,3), boxes(i,4));
end
fclose(fid);
end
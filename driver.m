%% driver file for supervised descent method
clear all; close all;

if 1
    tropts.path = 'C:\Users\Peihong\Desktop\Data\Detection\lfpw\trainset\';
    tropts.prefix = 'image_';
    tropts.imgext = '.png';
    tropts.ptsext = '.pts';
    tropts.digits = 4;
    tropts.npts = 68;
    tropts.nimgs = 871;
else
    trainset_path = 'C:\Users\Peihong\Desktop\Data\Detection\complete\trainingset\';
    listings = dir(trainset_path);
    inputfiles = cell(floor((numel(listings)-2)/2), 2);
    ninputs = 0;
    for i=1:numel(listings)
        if listings(i).isdir
            continue;
        end
        if strfind(listings(i).name, '.pts')
            continue;
        end
        if strfind(listings(i).name, '.box')
            continue;
        end
        ninputs = ninputs + 1;
        inputfiles{ninputs, 1} = [trainset_path, listings(i).name];
        inputfiles{ninputs, 2} = regexprep(inputfiles{ninputs, 1}, '\.\w+$', '.pts');
    end   
    tropts.inputfiles = inputfiles;
    tropts.nimgs = size(inputfiles, 1);
    tropts.npts = 68;
end

tropts.params.window_size = 128.0;
tropts.params.feat_window_size = 32;
tropts.params.nbins = 18;
tropts.params.oversample_rate = 1;
tropts.params.nstages = 5;

% train model
tic;
model = trainModel(tropts);
toc;

single_alignment;
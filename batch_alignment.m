%% batch alignment
close all;
%images_path = 'C:\Users\Peihong\Desktop\Data\Detection\random\';
%images_path = 'C:\Users\Peihong\Desktop\Data\Detection\complete\testset\';
images_path = 'C:\Users\Peihong\Desktop\Data\Detection\lfpw\testset\';
compute_error = true;

%model = load('model_3149_kappa_25_5_window_100.mat');
%model = load('model_3149_kappa_25_15_window_128.mat');
%model = load('model_4722_kappa_25_5_window_100.mat');

result_path = 'C:\Users\Peihong\Desktop\Code\Utilities\SupervisedDescentMethod\results\';
listings = dir(images_path);

supported_ext = {'.jpg', '.bmp', '.png', '.gif'};

valid_count = 0;
for i=1:numel(listings)
    if listings(i).isdir
        continue;
    end
    
    is_supported = false;
    for j=1:numel(supported_ext)
        if strfind(listings(i).name, supported_ext{j})
            is_supported = true;
            break;
        end
    end
    if ~is_supported
        continue;
    end
    
    imgfile = [images_path, listings(i).name];
    if compute_error
        ptsfile = [images_path, regexprep(listings(i).name, '\.\w+$', '.pts')];
        
        fid = fopen(ptsfile, 'r');
        textscan(fid, '%s', 3, 'Delimiter', '\n');
        points_ref = textscan(fid, '%f %f', 68, 'Delimiter', '\n');
        points_ref = cell2mat(points_ref);
        fclose(fid);        
    end
    
    img = imread(imgfile);
    [box, points, succeeded] = applyModel_batch(img, model);
    if succeeded
        figure;
        showImageWithPoints(img, box, points, points_ref);
        z=getframe(gcf);
        imwrite(z.cdata, [result_path, 'aligned_', listings(i).name]);
        close;
        for j=1:numel(points)
            mean_error = mean(sqrt(sum((points_ref - points{j}).^2, 2)));
            ref_dist = norm(mean(points_ref([38 39 41 42],:)) - mean(points_ref([44 45 47 48],:)));
            relative_error = mean_error / ref_dist
            if relative_error < 0.5 % should smaller than 0.5
                valid_count = valid_count + 1;
                error(valid_count) = relative_error;
                break;
            end
        end
    end
end
cdfplot(error);
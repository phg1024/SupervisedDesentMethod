%% perform alignment on a single image
% apply model to test data
[filename, pathname] = uigetfile({'*.jpg; *.png; *.gif; *.bmp'; '*.*'}, 'Choose Image');
img = imread([pathname, filename]);
tic;
[box, points, succeeded] = applyModel_batch(img, model);
toc;

if succeeded
    ptsfile = [pathname, regexprep(filename, '\.\w+$', '.pts')];
    if exist(ptsfile, 'file')
        fid = fopen(ptsfile, 'r');
        textscan(fid, '%s', 3, 'Delimiter', '\n');
        points_ref = textscan(fid, '%f %f', 68, 'Delimiter', '\n');
        points_ref = cell2mat(points_ref);
        fclose(fid);
    else
        points_ref = [];
    end
    
    figure;showImageWithPoints(img, box, points, points_ref);
    if ~isempty(points_ref)
        for j=1:numel(points)
            per_point_error = sqrt(sum((points_ref - points{j}).^2, 2));
            mean_error = mean(per_point_error);
            ref_dist = norm(mean(points_ref([38 39 41 42],:)) - mean(points_ref([44 45 47 48],:)));
            relative_error = mean_error / ref_dist;
            if relative_error < 0.5 % should smaller than 0.5
                relative_error
            end
        end
    end
else
    disp('no face found');
end
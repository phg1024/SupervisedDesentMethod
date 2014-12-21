function showImageWithPoints(img, boxes, pts, pts_ref)
imshow(img, 'border', 'tight');
hold on;
for i=1:numel(boxes)
    rectangle('Position', boxes{i}, 'EdgeColor', 'r');

    plot(pts{i}(:,1)-2, pts{i}(:,2)-2, 'og', 'MarkerSize', 4);
    if 0
        for j=1:size(pts{i},1)
            text(pts{i}(j,1), pts{i}(j,2), num2str(j));
        end
    end
end

if nargin > 3
    if ~isempty(pts_ref)        
        plot(pts_ref(:,1)-2, pts_ref(:,2)-2, 'xr', 'MarkerSize', 4);
    end
end

end
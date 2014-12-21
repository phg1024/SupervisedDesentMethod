function box = findValidBox(boxes, points)
npts = size(points, 1);
box = [];
% at least 3/4 points in the box
for i = 1:size(boxes, 1)
    b = boxes(i,:);
    xl = b(1); xr = b(1)+b(3)-1;
    yu = b(2); yd = b(2)+b(4)-1;
    cnt = npts;
    for j=1:npts
        px = points(j,1); py = points(j, 2);
        if px < xl || px > xr || py < yu || py > yd
            cnt = cnt-1;
        end
    end
    
    if cnt > npts*0.75
        box = boxes(i,:);
        return;
    end
end

end
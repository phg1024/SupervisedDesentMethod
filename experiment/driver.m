close all;

steps = [0.25, 0.025, 0.0025, 0.00025];
terror = zeros(length(steps), 1);
c = 0;
for sid = 1:length(steps)
    step = steps(sid);
    y0 = [-1:step:1]';
    x0 = asin(y0);
    h = @(x) sin(x);
    ystar = [-1:0.05:1]';
    
    n = 10;
    
    % train model
    R = cell(n,1);      x = cell(n, 1);
    phi = cell(n, 1);   error = zeros(n, 1);
    x{1} = repmat(c, length(x0), 1);
    error(1) = norm(x{1} - x0);
    for k=2:n
        dx = x0 - x{k-1};
        phi{k-1} = h(x{k-1});
        dphi = phi{k-1} - y0;
        R{k-1} = (dphi'*dphi)\(dphi'*dx);
        b{k-1} = mean(dx) - R{k-1} * mean(y0);
        x{k} = x{k-1} + R{k-1} * dphi;
        error(k) = norm(x{k} - x0);
    end
    
%     figure; hold on;
%     plot(x0, y0, '-x');
%     plot(x{n}, y0, '-o');
%     
%     figure; plot(error);
    
    % test it
    xstar = cell(n, 1);
    xstar{1} = repmat(c, length(ystar), 1);
    for k=2:n
        xstar{k} = xstar{k-1} + R{k-1} * (h(xstar{k-1}) - ystar) + b{k-1};
    end
    diffx = xstar{n} - asin(ystar);
    diffx = diffx(2:end-1);
    terror(sid) = sqrt((diffx'*diffx)/length(xstar))
    figure; hold on;
    plot(asin(ystar), ystar, '-x');
    plot(xstar{n}, ystar, '-o');
end

figure; plot(log(steps), terror, '-x');
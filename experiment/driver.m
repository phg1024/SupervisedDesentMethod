close all;

testCase = 3;

switch testCase
    case 1
        steps = [0.25, 0.025, 0.0025, 0.00025];
    case 2
        steps = [3, 1, 0.5];
    case 3
        steps = [3, 1, 0.5];
    case 4
        steps = [0.11, 0.055, 0.011];
end

terror = zeros(length(steps), 1);
c = 0;
for sid = 1:length(steps)
    step = steps(sid);
    
    switch testCase
        case 1
            h = @(x) sin(x);
            g = @(x) asin(x);
            y0 = [-1:step:1]';
            x0 = g(y0);
            ystar = [-1:0.05:1]';
        case 2
            h = @(x) exp(x);
            g = @(x) log(x);    
            y0 = [1:step:28]';
            x0 = g(y0);
            ystar = [1:0.5:28]';
        case 3
            h = @(x) x.^3;
            g = @(x) nthroot(x, 3);
            y0 = [-27:step:27]';
            x0 = g(y0);
            ystar = [-27:0.5:27]';
        case 4
            h = @(x) erf(x);
            g = @(x) erfinv(x);
            y0 = [-0.99:step:0.99]';
            x0 = g(y0);
            ystar = [-0.99:0.03:0.99]';
    end
    
    n = 10;
    
    % train model
    R = cell(n,1);      x = cell(n, 1);
    phi = cell(n, 1);   error = zeros(n, 1);
    x{1} = repmat(c, length(x0), 1);
    error(1) = norm(x{1} - x0);
    for k=2:n
        dx = x0 - x{k-1};
        phi{k-1} = h(x{k-1});
        dphi = y0 - phi{k-1};
        R{k-1} = (dphi'*dphi)\(dphi'*dx);
        b{k-1} = mean(dx) - R{k-1} * mean(dphi);
        x{k} = x{k-1} + R{k-1} * dphi;
        error(k) = norm(x{k} - x0);
    end
    
    figure; hold on;
    plot(x0, y0, '-gx');
    plot(x{n}, y0, '-ro');
    title('training');
    
    figure; plot(error);
    title('error');
    
    % test it
    xstar = cell(n, 1);
    xstar{1} = repmat(c, length(ystar), 1);
    for k=2:n
        xstar{k} = xstar{k-1} + R{k-1} * (ystar - h(xstar{k-1})) + b{k-1};
    end
    diffx = xstar{n} - g(ystar);
    diffx = diffx(2:end-1);
    terror(sid) = sqrt((diffx'*diffx)/length(xstar))
    figure; hold on;
    plot(g(ystar), ystar, '-gx');
    plot(xstar{n}, ystar, '-ro');
    title('test');
end

figure; plot(log(steps), terror, '-x');
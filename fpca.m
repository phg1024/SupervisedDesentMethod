function [coeffs, explained] = fpca(X)
covX = cov(X);
if 0
    [coeffs, ~, explained] = pcacov(covX);
else
    %[~, S, coeffs] = svd(covX);
    [coeffs, explained] = eig(covX);
    [explained, order] = sort(diag(explained),'descend');  %# sort eigenvalues in descending order
    coeffs = coeffs(:,order);
end

explained = abs(explained) / sum(explained) * 100.0;
end
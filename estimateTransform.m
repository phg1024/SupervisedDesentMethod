function [s, R, t] = estimateTransform(p, q)
[n, m] = size(p);

mu_p = mean(p);
mu_q = mean(q);

dp = p - repmat(mu_p, n, 1);
sig_p2 = sum(sum(dp .* dp))/n;

dq = q - repmat(mu_q, n, 1);
sig_q2 = sum(sum(dq .* dq))/n;

sig_pq = dq' * dp / n;

det_sig_pq = det(sig_pq);
S = diag(ones(m, 1));
if det_sig_pq < 0
    S(n, m) = -1;
end

[U, D, V] = svd(sig_pq);

R = U * S * V';
s = trace(D*S)/sig_p2;
t = mu_q' - s * R * mu_p';
end
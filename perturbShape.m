function shape = perturbShape(S0, box, smean, svar, tmean, tvar)
s = clamp(normrnd(smean, svar), 0.8, 1.2);
t = clamp(normrnd(tmean, tvar), -0.10, 0.10) * box(3);

Lfp = length(S0); Nfp = Lfp/2;

S = reshape(S0, Nfp, 2);
mS = mean(S);
S1 = s * (S - repmat(mS, Nfp, 1)) + repmat(mS + t, Nfp, 1);

shape = reshape(S1, 1, Lfp);

end

function v = clamp(v, lower, upper)
v = min(max(v, lower), upper);
end
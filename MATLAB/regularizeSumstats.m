function z_regularized = regularizeSumstats(P,z,whichIndices,lambdaParam)
%regularizeSumstats alters a vector of Z scores or marginal effect-size
%estimates (in per-sd units) to make them conform with the LD patterns of
%an LDGM precision matrix P
%   Detailed explanation goes here

noIndices = cellfun(@length, P);
x = precisionMultiply(P,z,whichIndices);
x = cellfun(@assignto, x, whichIndices, num2cell(noIndices), 'UniformOutput', false);
y = cellfun(@mtimes, P, x, 'UniformOutput', false);

I_lambda_P = cellfun(@(p)lambdaParam*speye(size(p)) + p, ...
    P, 'UniformOutput',false);
y = cellfun(@mldivide,I_lambda_P,y,'UniformOutput',false);
y = cellfun(@(x,j)x(j), y, whichIndices, 'UniformOutput', false);

%P^-1 * (1-lambda)*P^-1 + lambda*I) \ z
z_regularized = precisionDivide(P,y,whichIndices);

end
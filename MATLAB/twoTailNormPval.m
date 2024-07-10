function [pval] = twoTailNormPval(params, paramsVar)
    % two-tailed p-value for the estimates
    % Variance is specified as a matrix

    SE = sqrt(diag(paramsVar));
    Z_stat = params ./ SE; 
    pval = 2 * normcdf(abs(Z_stat), 'upper');
end

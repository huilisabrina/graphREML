function [pval] = compute_pval(params, paramsVar)
    SE = sqrt(diag(paramsVar));
    Z_stat = params ./ SE; 
    pval = 2 * normcdf(abs(Z_stat), 'upper');
end

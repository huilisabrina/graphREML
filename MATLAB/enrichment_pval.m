function [pval] = enrichment_pval(h2, h2SE, cov, p)
    T = h2 ./ p - h2(1) / p(1);
    var_vec = diag(cov)';
    SE = sqrt(var_vec ./ (p.^2) + var_vec(1) ./ (p(1)^2) - ...
        2 * var_vec(1,:) ./ p / p(1));
    Z_stat = T ./ SE;
    Z_stat(1) = h2(1) / h2SE(1);
    pval = 2 * normcdf(abs(Z_stat), 'upper');
end
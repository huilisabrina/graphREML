function [pval] = enrichment_pval(h2, h2SE, h2Cov, p_vec, varargin)
% Compute p-values for the enrichment (test on the difference of ratios)
%   Required inputs:
%   h2: vector of heritability, including total
%   h2SE: vector of SE for heritability
%   h2Cov: covariance matrix of h2
%   p_vec: vector of annotations sizes
%   refCol: index of the annotation to compute enrichment

    p=inputParser;
    addRequired(p, 'h2', @(x)isvector(x));
    addRequired(p, 'h2SE', @(x)isvector(x));
    addRequired(p, 'h2Cov', @(x)ismatrix(x));
    addRequired(p, 'p_vec', @(x)isvector(x));
    addOptional(p, 'refCol', 1, @isnumeric);
    parse(p, h2, h2SE, h2Cov, p_vec, varargin{:});

    % turns p.Results.x into just x
    parse(p, sumstats_fp, varargin{:});
    fields = fieldnames(p.Results);
    for k=1:numel(fields)
        line = sprintf('%s = p.Results.%s;', fields{k}, fields{k});
        eval(line);
    end

    datetime.setDefaultFormats('default','yyyy-MM-dd hh:mm:ss')
    
    T = h2 ./ p_vec - h2(refCol) / p_vec(refCol);
    var_vec = diag(h2Cov)';
    SE = sqrt(var_vec ./ (p_vec.^2) + var_vec(refCol) ./ (p_vec(refCol)^2) - ...
        2 * var_vec(refCol,:) ./ p_vec / p_vec(refCol));
    Z_stat = T ./ SE;
    Z_stat(1) = h2(1) / h2SE(1);
    pval = 2 * normcdf(abs(Z_stat), 'upper');

end

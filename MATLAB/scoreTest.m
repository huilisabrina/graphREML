function [score, scoreTestStat, chisq_pval] = scoreTest(Z,P,null_params,null_cols,test_col,varargin)
%h2newtoncomputes Newton-Raphson maximum-likelihood heritability estimates
%   Required inputs:
%   Z: Z scores, as a cell array with one cell per LD block.
%   P: LDGM precision matrices as a cell array.
%   null_params: parameter fit from the baseline LD model
%   null_cols: columns of the annot matrix that are used for the null
%   test_col: column of the annot matrix to test for new significance testing
%
%   Optional inputs:
%   sampleSize: GWAS sample size
%   whichIndicesSumstats: cell array with one entry per LD block,
%   indicating which rows/cols in the precision matrix correspond to the 
%   summary statistics. Recommended to use as many SNPs as possible. 
%   Highly recommended to compute this using mergesnplists.m
%   There should be no duplicates (SNPs with the same index)
%   whichIndicesAnnot: which rows/cols of each precision matrix correspond 
%   to the rows of the annotation matrices. There can be SNPs with
%   duplicate indices.
%   annot: annotation matrices, one for each LD block, same number of SNPs
%   as Z
%   linkFn: function mapping from annotation vectors to per-SNP h2
%   linkFnGrad: derivative of linkFn
%   noSamples: number of samples for approximately computing trace of an
%   inverse matrix when calculating gradient. Default 0 uses exact
%   calculation instead of sampling.
%   printStuff: verbosity
%
%   Output arguments:
%   score: score function of the full likelihood, evaluated at the null
%   scoreTestStat: score test statistics, using jackknife covariance
%   chisq_pval: p-value from a chi-sq test


% initialize
p=inputParser;
if iscell(P)
    mm = sum(cellfun(@length,P));
else
    mm = length(P);
end

% required arguments
addRequired(p, 'Z', @(x)isvector(x) || iscell(x))
addRequired(p, 'P', @(x)ismatrix(x) || iscell(x))
addRequired(p, 'null_params', @(x)isvector(x) || iscell(x))
addRequired(p, 'null_cols', @(x)isvector(x) || iscell(x))
addRequired(p, "test_col", @isscalar)

% optional arguments
addOptional(p, 'sampleSize', 1, @isscalar)
addOptional(p, 'whichIndicesSumstats', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'whichIndicesAnnot', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'annot', cellfun(@(x){true(size(x))},Z), @(x)size(x,1)==mm || iscell(x))
addOptional(p, 'linkFn', @(a,x)log(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'linkFnGrad', @(a,x)exp(a*x).*a./(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'printStuff', true, @isscalar)
addOptional(p, 'noSamples', 0, @(x)isscalar(x) & round(x)==x)


% parse parameters
parse(p, Z, P, null_params, null_cols, test_col, varargin{:});

% turns p.Results.x into just x
fields = fieldnames(p.Results);
for k=1:numel(fields)
    line = sprintf('%s = p.Results.%s;', fields{k}, fields{k});
    eval(line);
end

blocksize = cellfun(@length, Z);

% extract annotation columns for null and for testing
annot_null = cellfun(@(x)x(:, null_cols), annot,'UniformOutput',false)
annot_test = cellfun(@(x)x(:, test_col), annot,'UniformOutput',false)
annot = cellfun(@(x,y)[x,y], annot_null, annot_test,'UniformOutput',false)

% count the number of annotations
noAnnot = size(annot{1},2);
annotSum = cellfun(@(x){sum(x,1)},annot);
annotSum = sum(vertcat(annotSum{:}));
noBlocks = length(annot);

% concatenate parameter vector for testing
params = null_params;
params(end+1) = 0;

if isempty(params)
    params = zeros(noAnnot,1);
end
smallNumber = 1e-6;

noParams = length(params)
annot_unnormalized = annot;
annot = annot;
% annot = cellfun(@(a){mm*a./max(1,annotSum)},annot);


% samples to be used to approximate the gradient
if noSamples > 0
    samples = arrayfun(@(m)randn(m,noSamples),blocksize,'UniformOutput',false);
    samples = cellfun(@(x)x./sqrt(sum(x.^2,2)),samples,'UniformOutput',false);
else
    samples = cell(noBlocks,1); % needed for parfor
end

% needed for parfor
linkFn = linkFn;
linkFnGrad = linkFnGrad; %#ok<*NODEF>
sampleSize = sampleSize;
noSamples = noSamples;

% Merging between annotations + sumstats
r2_proxy = struct('oldIndices',cell(noBlocks,1), 'newIndices', cell(noBlocks,1), 'r2', cell(noBlocks,1));
whichSumstatsAnnot = cell(noBlocks,1);

for block = 1:noBlocks
    [uniqueIndices, ~, duplicates] = unique(whichIndicesAnnot{block});
    
    % Handle missingness in the annotations matrix
    isMissing = ~ismember(whichIndicesSumstats{block}, uniqueIndices);
    if any(isMissing)
        whichIndicesSumstats{block} = whichIndicesSumstats{block}(~isMissing);
        Z{block} = Z{block}(~isMissing);
    end
    
    % Handle missingness in the summary statistics
    isMissing = ~ismember(uniqueIndices, whichIndicesSumstats{block});
    if any(isMissing)
        identity = speye(length(uniqueIndices));
        R_missing = precisionDivide(P{block}, identity(:,isMissing), uniqueIndices);
        R_missing(isMissing,:) = 1i; % Make sure never to pick a missing SNP as proxy
        r2_proxy(block).oldIndex = uniqueIndices(isMissing);
        [r2_proxy(block).r2, idx] = max(R_missing.^2);
        r2_proxy(block).newIndex = uniqueIndices(idx);
        uniqueIndices(isMissing) = r2_proxy(block).newIndex;
        whichIndicesAnnot{block} = uniqueIndices(duplicates);
    end

    % Re-order sumstats so indices are sorted
    [whichIndicesSumstats{block}, reordering] = sort(whichIndicesSumstats{block});
    Z{block} = Z{block}(reordering);
    
    % Mapping from annotations matrices to SNPs in the sumstats
    [~, ~, whichSumstatsAnnot{block}] = unique(whichIndicesAnnot{block});
end

% compute and record gradient for each block
scoreStat_block = zeros(noBlocks, noParams);

parfor block = 1:noBlocks
    % Effect-size variance for each sumstats SNPs, adding across annotations
    sigmasq = accumarray(whichSumstatsAnnot{block}, ...
        linkFn(annot{block}, params(1:noParams)));
    
    % Gradient of the effect-size variance for each sumstats SNP
    sg = linkFnGrad(annot{block}, params(1:noParams));
    sigmasqGrad = zeros(length(sigmasq),size(sg,2));

    for kk=1:size(sg,2)
        sigmasqGrad(:,kk) = accumarray(whichSumstatsAnnot{block}, sg(:,kk));
    end
    
    % Gradient of the log-likelihood
    if noSamples > 0
        scoreStat_block(block,:) = GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, samples{block})';
    else
        scoreStat_block(block,:) = GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, 1, 1)';
    end
end

% construct score test
score = sum(scoreStat_block,1);

for block = 1:noBlocks
    psudoJkScore(block,:) = score - scoreStat_block(block,:);
end

% compute the empirical covariance using jackknife
jkVarScore = cov(psudoJkScore) * (noBlocks-2);
scoreTestStat = score * pinv(jkVarScore) * transpose(score) ;
[h,chisq_pval] = chi2gof(scoreTestStat,'Alpha',0.05);
chisq_pval =  chi2cdf(scoreTestStat, 1, 'upper');

end

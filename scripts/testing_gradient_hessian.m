% This script loads LDGMs, simulates some GWAS summary statistics, and
% calculates the likelihood, its gradient, and its hessian. It empirically
% verifies that the gradient and hessian are correct.
% 
% Script requires that the ldgm repo is cloned in the parent directory of h2-ldgm
% Script requires that in parent of parent directory, there is a
% subdirectory 'data' that contains 'LDGMs_1kg'

clear;clc
addpath(genpath('../../ldgm/MATLAB'))% add MATLAB directory and its subdirectories to path
addpath(genpath('..'))

populations = {'EUR'};
chromosome = 22;
ldgms_dir = ['../../../data/LDGMs_1kg/ldgm/1kg_chr',num2str(chromosome),'_'];

% load precision matrices and SNP lists
[P, snplists] = loadLDGMs(ldgms_dir,populations);
[noBlocks, noPopns] = size(P);
blockSize = cellfun(@length,P);

% Extract allele frequencies from SNP lists
AF = cell(noBlocks,noPopns);
for ii = 1:noBlocks
    % call unique() on the SNP list indices with two output arguments in
    % order to pick a representative SNP for each index
    [~,representatives] = unique(snplists{ii}.index,'stable');

    % snplists table can be sliced using the names of each column as a cell
    % array of strings
    AF(ii,:) = mat2cell(table2array(snplists{ii}(representatives,populations)),...
        length(representatives),ones(1,noPopns));
end

% simulation parameters
h2true = 0.1;
sampleSize = sum(blockSize) / 5; 

% True annotations: assign every other LD block to one annotation or the
% other
annotations = cell(noBlocks,1);
for block = 1:noBlocks
    annotations{block} = ones(blockSize(block),2) .* (mod(block,2) == [0 1]) ;
end

% Stratified heritability: all in one annotation
heritability = [h2true 0];

% simulate summary statistics
[sumstats, whichIndices, beta_perallele, beta_perSD] = ...
    simulateSumstats(sampleSize, 'alleleFrequency', AF,...
    'precisionMatrices',P,...
    'annotations', annotations,...
    'heritability',heritability,...
    'missingness',0);

% extract Z scores from sumstats tables
Z = cellfun(@(s){s.Z_deriv_allele},sumstats);

% heterozygosity = 2pq
heterozygosity = cellfun(@(af,ii)2*af(ii).*(1-af(ii)),AF,whichIndices,'UniformOutput',false);

% calculate likelihood, its gradient, and its hessian for var(beta) in a
% grid, w.r.t. x
x = mean(vertcat(beta_perallele{:}).^2);
grid = x * [1 1+1e-6];
for ii = 1:length(grid)
    % normalized effect-size variance of each SNP
    sigmasq = cellfun(@(S){grid(ii)*S},heterozygosity);
    
    % log density
    tic;
    likelihood(ii) = GWASlikelihood(Z,sigmasq,P,sampleSize,whichIndices);
    time_likelihood(ii) = toc;
    
    % d/dx log density
    tic;
    grad_cells = GWASlikelihoodGradient(Z,sigmasq,P,sampleSize,heterozygosity,whichIndices);
    grad(ii) = sum([grad_cells{:}]);
    time_grad(ii) = toc;
    
    % d2/dx2 log density
    tic;
    hessian_cells = GWASlikelihoodHessian(Z,sigmasq,P,sampleSize,heterozygosity,whichIndices);
    hess(ii) = sum([hessian_cells{:}]);
    time_hess(ii) = toc;
end

% 1st derivative should agree with difference b/t likelihoods
disp([grad(1) (likelihood(2) - likelihood(1))/(grid(2)-grid(1))])

% 2nd derivative should agree with difference b/t gradients
disp([hess(1) (grad(2) - grad(1))/(grid(2)-grid(1))])

% use the "true" annotations multiplied by the heterozygosity
annot = cellfun(@(a,j,b){[a(j,:) a(j,:).*b]},annotations, whichIndices, heterozygosity);

% estimate heritability
tic;
[h2,steps] = h2newton(Z,P,'whichIndices',whichIndices,...
    'nn',sampleSize,...
    'annot',annot,...
    'noSamples',0);

time_h2newton = toc;

% first two heritability estimates should be around h2_true, 0
disp([h2.h2(1:2)])

beep






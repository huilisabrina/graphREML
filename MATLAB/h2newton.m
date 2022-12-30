function [estimate, steps] = h2newton(Z,P,varargin)
%h2newtoncomputes Newton-Raphson maximum-likelihood heritability estimates
%   Required inputs:
%   Z: Z scores, as a cell array with one cell per LD block.
%   P: LD graphacal model as a cell array.
% 
%   Optional inputs:
%   nn: GWAS sample size
%   whichIndices: which rows/cols in the LDGM have nonmissing summary statistics.
%   Recommended to use as many SNPs as possible. All SNPs in the summary
%   statistics vectors should be in the LDGM.
%   annot: annotation matrices, one for each LD block, same number of SNPs
%   as alphahat
%   linkFn: function mapping from annotation vectors to per-SNP h2
%   linkFnGrad: derivative of linkFn
%   params: initial point for parameters
%   noSamples: number of samples for approximately computing trace of an
%   inverse matrix when calculating gradient. Default 0 uses exact
%   calculation instead of sampling.
%   printStuff: verbosity
%   fixedIntercept: whether intercept should be fixed at 1/nn, or learned
%   automatically. Default 1. fixedIntercept=0 currently unsupported.
%   convergenceTol: terminates when objective function improves less than
%   this
%   convergenceTol: when log-likelihood improves by less, declare convergence
%   maxReps: maximum number of steps to perform
%   minReps: starts checking for convergence after this number of steps
% 
%   Output arguments:
%   estimate: struct containing the following fields:
%       params: estimated paramters
%       h2: heritability estimates for each annotation; equal to \sum_j
%       h^2(j) a_kj, where a_kj is annotation value for annotation k and
%       SNP j, and h^2(j) is the per-SNP h2 of SNP j
%       annotSum: sum of annotation values for each annotation across SNPs
%       logLikelihood: likelihood of sumstats at optimum
%       nn: sample size or 1/intercept if intercept was learned
%       paramsVar: approx sampling covariance of the parameters
%       h2Var: approx sampling covariance of the heritability
%       h2SE: approx standard error of the heritability
%   
%   steps: struct containing the following fields:
%       reps: number of steps before stopping
%       params: parameter values at each step
%       obj: objective function value (-logLikelihood) at each step
%       gradient: gradient of the objective function at each step
% 

% initialize
p=inputParser;
if iscell(P)
    mm = sum(cellfun(@length,P));
else
    mm = length(P);
end

addRequired(p, 'alphahat', @(x)isvector(x) || iscell(x))
addRequired(p, 'P', @(x)ismatrix(x) || iscell(x))
addOptional(p, 'sampleSize', 1, @isscalar)
addOptional(p, 'whichIndices', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'annot', cellfun(@(x){true(size(x))},Z), @(x)size(x,1)==mm || iscell(x))
addOptional(p, 'linkFn', @(x,w)max(x*w,0), @(f)isa(f,'function_handle'))
addOptional(p, 'linkFnGrad', @(x,w)x.*(x*w>=0), @(f)isa(f,'function_handle'))
addOptional(p, 'params', [], @isvector)
addOptional(p, 'fixedIntercept', true, @isscalar)
addOptional(p, 'printStuff', true, @isscalar)
addOptional(p, 'noSamples', 0, @(x)isscalar(x) & round(x)==x)
addOptional(p, 'convergenceTol', 1e-6, @isscalar)
addOptional(p, 'maxReps', 1e2, @isscalar)
addOptional(p, 'minReps', 1, @isscalar)
addOptional(p, 'stepSizeParam', 1e-3, @isscalar)

parse(p, Z, P, varargin{:});

% turns p.Results.x into just x
fields = fieldnames(p.Results);
for k=1:numel(fields)
    line = sprintf('%s = p.Results.%s;', fields{k}, fields{k});
    eval(line);
end

blocksize = cellfun(@length, alphahat);

noAnnot = size(annot{1},2);
annotSum = cellfun(@(x){sum(x,1)},annot);
annotSum = sum(vertcat(annotSum{:}));

noBlocks = length(annot);
if isempty(params)
    params = zeros(noAnnot,1);
    if ~fixedIntercept
        params(end+1) = 1/sampleSize;
    end
end
smallNumber = 1e-12;
noParams = length(params) - 1 + fixedIntercept;

annot_unnormalized = annot;
annot = cellfun(@(a){mm*a./max(1,annotSum)},annot);
annotCat = vertcat(annot{:});

if fixedIntercept
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params)}, annot),...
        P, sampleSize, whichIndices);
    
else
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params(1:end-1))}, annot),...
        P, 1/max(smallNumber, params(end)), whichIndices);
    
end

newObjVal = objFn(params);

% samples to be used to approximate the gradient
if noSamples > 0
    samples = arrayfun(@(m)randn(m,noSamples),blocksize,'UniformOutput',false);
    samples = cellfun(@(x)x./sqrt(sum(x.^2,2)),samples,'UniformOutput',false);
else
    samples = cell(noBlocks,1); % needed for parfor
end

% needed for parfor
whichIndices = whichIndices; %#ok<*ASGSL> 
linkFn = linkFn;
linkFnGrad = linkFnGrad; %#ok<*NODEF> 
sampleSize = sampleSize;
fixedIntercept = fixedIntercept;
noSamples = noSamples;

allSteps=zeros(min(maxReps,1e6),noParams+1-fixedIntercept);
allValues=zeros(min(maxReps,1e6),1);
allGradients=allSteps;

% main loop
for rep=1:maxReps
    if printStuff
        disp(rep)
    end
    
    % Compute gradient and hessian by summing over blocks
    gradient = 0;
    hessian = 0;
    for block = 1:noBlocks
        sigmasq = linkFn(annot{block}, params(1:noParams));
        sigmasqGrad = linkFnGrad(annot{block}, params(1:noParams));
        
        hessian = hessian + ...
            GWASlikelihoodHessian(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndices{block}, fixedIntercept)';
        
        if noSamples > 0
            gradient = gradient + ...
                GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndices{block}, samples{block})';
        else
            gradient = gradient + ...
                GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndices{block}, fixedIntercept)';
        end
    end
    
    % Compute step
    params = params - (hessian + stepSizeParam * diag(diag(hessian)) + 1e-12 * eye(length(params))) \ gradient;
    
    % New objective function value
    newObjVal = objFn(params);
    
    allValues(rep)=newObjVal;
    if nargout > 1
        allSteps(rep,:)=params;
        %         allStepSizes(rep,:) = stepSizeVector;
    end
    if rep > 1
        if allValues(rep-1) - newObjVal < 0
            warning('Objective function increased at the last iteration')
            break;
        end
        if allValues(rep-1) - newObjVal < convergenceTol
            break;
        end
    end
    
    %     converged = (newObjVal-oldObjVal)/oldObjVal < convergenceTol;
end

% Convergence report
if nargout > 1
    steps.reps = rep;
    steps.params = allSteps(1:rep,:);
    steps.obj = allValues(1:rep);
    steps.gradient = allGradients(1:rep,:);
end

% From paramaters to h2 estimates
h2Est = 0;
for block=1:noBlocks
    perSNPh2 = linkFn(annot{block}, params(1:noParams));
    h2Est = h2Est + sum(perSNPh2.*annot_unnormalized{block});
end

estimate.params = params';
estimate.h2 = h2Est;
estimate.annotSum = annotSum;
estimate.logLikelihood = -newObjVal;

% Enrichment only calculated if first annotation is all-ones vector
if all(cellfun(@(a)all(a(:,1)==1),annot_unnormalized))
    estimate.enrichment = (h2Est./annotSum) / (h2Est(1)/annotSum(1));
end

%% Compute covariance of the parameter estimates
% Approximate fisher information (actually average information)
FI = -hessian;

if any(diag(FI)==0)
    warning('Some parameters have zero fisher information. Regularizing FI matrix to obtain standard errors.')
    FI = FI + smallNumber*eye(size(FI));
end

% Variance of h2 estimates via chain rule
dh2da = 0;
for block=1:noBlocks
    dh2da = dh2da + linkFnGrad(annot{block}, params(1:noParams))'*annot_unnormalized{block};
end
h2Var = dh2da' * (FI(1:noParams, 1:noParams) \ dh2da);

estimate.paramsVar = pinv(FI);
estimate.h2Var = h2Var;
estimate.h2SE = sqrt(diag(h2Var))';

if ~fixedIntercept
    estimate.intercept = params(end) * sampleSize;
    estimate.interceptSE = sqrt(estimate.paramsVar(end,end)) * sampleSize;
else
    estimate.intercept = 1;
    estimate.interceptSE = 0;
end

% estimate.interceptSE = sqrt(estimate.paramsVar(end,end));
% if all(cellfun(@(a)all(a(:,1)==1),annot_unnormalized))
%     estimate.enrichmentZscore = (h2Est./annotSum - h2Est(1)/annotSum(1)) ./ ...
%         sqrt(diag(h2Var)'./annotSum.^2 + h2Var(1)./annotSum(1).^2 - 2*h2Var(1,:)/annotSum(1)./annotSum);
%     estimate.enrichmentZscore(1) = 0;
% end


end


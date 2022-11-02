function [estimate, steps, objFn] = h2newton(Z,P,varargin)
%h2newtoncomputes Newton-Raphson maximum-likelihood heritability estimates
%   Inputs:
%   alphahat: effect-size estimates, units of sd(Y)/sd(X), or Z scores.
%   Should be a cell array with one cell per LD block.
%   P: LD graphacal model as a cell array.
%   nn: GWAS sample size
%   whichSNPs: which SNPs in the LDGM have nonmissing summary statistics.
%   Recommended to use as many SNPs as possible. All SNPs in the summary
%   statistics vectors should be in the LDGM.
%   annot: annotation matrices, one for each LD block, same number of SNPs
%   as alphahat
%   linkFn: function mapping from annotation vectors to per-SNP h2
%   linkFnGrad: derivative of linkFn
%   params: initial point for parameters
%   convergenceTol: terminates when objective function improves less than
%   this
%   maxReps: maximum number of steps to perform
%   minReps: starts checking for convergence after this number of steps
%   method: please use 'gradient'

% initialize
p=inputParser;
if iscell(P)
    mm = sum(cellfun(@length,P));
else
    mm = length(P);
end

addRequired(p, 'alphahat', @(x)isvector(x) || iscell(x))
addRequired(p, 'P', @(x)ismatrix(x) || iscell(x))
addOptional(p, 'nn', 1e3, @isscalar)
addOptional(p, 'whichSNPs', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'annot', cellfun(@(x){true(size(x))},Z), @(x)size(x,1)==mm || iscell(x))
addOptional(p, 'linkFn', @(a,x)max(a*x,0), @(f)isa(f,'function_handle'))
addOptional(p, 'linkFnGrad', @(a,x)a.*(a*x>=0), @(f)isa(f,'function_handle'))
addOptional(p, 'params', [], @isvector)
addOptional(p, 'convergenceTol', 1e-6, @isscalar)
addOptional(p, 'maxReps', 1e2, @isscalar)
addOptional(p, 'minReps', 3, @isscalar)
addOptional(p, 'noVarianceEstimate', 1, @isscalar)
addOptional(p, 'fixedIntercept', true, @isscalar)
addOptional(p, 'printStuff', true, @isscalar)
addOptional(p, 'stepSizeMethod', 'adagrad', @isstr)
addOptional(p, 'stepMemory', 10, @isscalar)
addOptional(p, 'momentumParam', 0, @(x)isscalar(x) & x<1 & x>=0)
addOptional(p, 'stepSizeFactor', 2, @(x)isscalar(x) & x>1)
addOptional(p, 'noSamples', 3, @(x)isscalar(x) & round(x)==x)

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
end
smallNumber = 1e-12;
noParams = length(params);

annot_unnormalized = annot;
annot = cellfun(@(a){mm*a./max(1,annotSum)},annot);
annotCat = vertcat(annot{:});

if fixedIntercept
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params)}, annot),...
        P, nn, whichSNPs);

else
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params(1:end-1))}, annot),...
        P, 1/max(smallNumber, params(end)), whichSNPs);
    params(end+1) = 1/nn;
end

newObjVal = objFn(params);

% samples to be used to approximate the gradient
if noSamples > 0
    samples = arrayfun(@(m)randn(m,noSamples),blocksize,'UniformOutput',false);
    samples = cellfun(@(x)x./sqrt(sum(x.^2,2)),samples,'UniformOutput',false);
end

gradient = zeros(noParams + 1 - fixedIntercept,1);
stepSizeVector = ones(noParams + 1 - fixedIntercept,1)/nn/mm^2;

allSteps=zeros(min(maxReps,1e6),noParams+1-fixedIntercept);
allValues=zeros(min(maxReps,1e6),1);
allStepSizes=allSteps;
allGradients=allSteps;

for rep=1:maxReps
    if printStuff
        disp(rep)
    end

    % Compute gradient and hessian
    gradient = 0;
    hessian = 0;
    for block = 1:noBlocks
        sigmasq = linkFn(annot{block}, params(1:noParams));
        sigmasqGrad = linkFnGrad(annot{block}, params(1:noParams));

        hessian = hessian + ...
            GWASlikelihoodHessian(Z{block},sigmasq,P{block},...
            nn, sigmasqGrad, whichSNPs{block}, fixedIntercept)';

        if noSamples > 0
            gradient = gradient + ...
                GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                nn, sigmasqGrad, whichSNPs{block}, samples{block})';
        else
            gradient = gradient + ...
                GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                nn, sigmasqGrad, whichSNPs{block},true)';
        end
    end

    % Compute step
    params = params - (hessian + eps * eye(noParams)) \ gradient;

    % New objective function value
    newObjVal = objFn(params);

    allValues(rep)=newObjVal;
    if nargout > 1
        allSteps(rep,:)=params;
        %         allStepSizes(rep,:) = stepSizeVector;
    end
    if rep > minReps
        if allValues(rep-minReps) - newObjVal < convergenceTol * minReps
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
    steps.stepsize = allStepSizes(1:rep,:);
    steps.gradient = allGradients(1:rep,:);
end

% From paramaters to h2 estimates
h2Est = 0;
for block=1:noBlocks
    perSNPh2 = linkFn(annot{block}, params(1:noParams));
    h2Est = h2Est + sum(perSNPh2.*annot_unnormalized{block});
end

estimate.params = params(1:noParams);
estimate.h2 = h2Est;
estimate.annotSum = annotSum;
estimate.likelihood = -newObjVal;
estimate.nn = nn;

% Enrichment only calculated if first annotation is all-ones vector
if all(cellfun(@(a)all(a(:,1)==1),annot_unnormalized))
    estimate.enrichment = (h2Est./annotSum) / (h2Est(1)/annotSum(1));
end

% Compute covariance of the parameter estimates
if ~noVarianceEstimate
    smallNumber = 1e-6;
    gradient = 0;gradient_k = 0;
    for block = 1:noBlocks
        sigmasq = linkFn(annot{block}, params(1:noParams));
        sigmasqGrad = linkFnGrad(annot{block}, params(1:noParams));
        gradient = gradient + ...
            GWASlikelihoodGradient(Z{block},1./sigmasq,P{block},nn,...
            sigmasqGrad,whichSNPs{block},fixedIntercept)';
        if ~fixedIntercept
            gradient_k = gradient_k + ...
                GWASlikelihoodGradient(Z{block},1./sigmasq,P{block},...
                1/(1/nn*(1+smallNumber)),...
                sigmasqGrad,whichSNPs{block},fixedIntercept)';
        end
    end

    FI = zeros(noParams+1-fixedIntercept);
    if ~fixedIntercept
        FI(:,end) = (gradient_k - gradient) / (smallNumber * 1/nn);
    end
    for k = 1:noParams
        newParams = params;
        newParams(k) = params(k) * (1+smallNumber);
        gradient_k = 0;
        for block = 1:noBlocks
            sigmasq = linkFn(annot{block}, newParams(1:noAnnot));
            sigmasqGrad = linkFnGrad(annot{block}, newParams(1:noAnnot));
            gradient_k = gradient_k + ...
                GWASlikelihoodGradient(Z{block},1./sigmasq,P{block},...
                nn,sigmasqGrad,whichSNPs{block},fixedIntercept)';
        end

        FI(:,k) = (gradient_k - gradient) / (smallNumber * params(k));
    end

    if any(diag(FI)==0)
        warning('Some parameters have zero fisher information. Regularizing FI matrix to obtain standard errors.')
        FI = FI + smallNumber*eye(size(FI));
    end


    % Variance of h2 estimates
    dh2da = 0;
    for block=1:noBlocks
        dh2da = dh2da + linkFnGrad(annot{block}, params(1:noParams))'*annot_unnormalized{block};
    end
    h2Var = dh2da'/FI(1:noParams,1:noParams) * dh2da;

    estimate.paramsVar = pinv(FI);
    estimate.fisherInfo = FI;
    estimate.h2Var = h2Var;
    estimate.h2SE = sqrt(diag(h2Var))';
    estimate.interceptSE = sqrt(estimate.paramsVar(end,end));

    if all(cellfun(@(a)all(a(:,1)==1),annot_unnormalized))
        estimate.enrichmentZscore = (h2Est./annotSum - h2Est(1)/annotSum(1)) ./ ...
            sqrt(diag(h2Var)'./annotSum.^2 + h2Var(1)./annotSum(1).^2 - 2*h2Var(1,:)/annotSum(1)./annotSum);
        estimate.enrichmentZscore(1) = 0;
    end


end

end

function [thetaNew, step, newObj] = linesearch(initTheta, initObj, grad, objFn, step, stepsize_factor)

oldObj = initObj;

if ~isreal(oldObj); error('objective function value should be real at initial point for line search'); end

step = step * stepsize_factor;

newObj = objFn(initTheta - step .* grad);

if newObj > oldObj
    while newObj > oldObj %-  sum(step .* grad.^2) / 10
        step = step / stepsize_factor;
        newObj = objFn(initTheta - step .* grad);
    end
else
    while newObj < oldObj %-  sum(step .* grad.^2) / stepsize_factor
        oldObj = newObj;
        step = step * stepsize_factor;
        newObj = objFn(initTheta - step .* grad);
    end
    newObj = oldObj;
    step = step / stepsize_factor;
end

thetaNew = initTheta - step .* grad;
end


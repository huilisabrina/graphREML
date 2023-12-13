function [estimate, steps, r2_proxy, diagnostics, jackknife] = run_graphREML(Z,P,varargin)
%h2newtoncomputes Newton-Raphson maximum-likelihood heritability estimates
%   Required inputs:
%   Z: Z scores, as a cell array with one cell per LD block.
%   P: LDGM precision matrices as a cell array.
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
%   params: initial point for parameters
%   noSamples: number of samples for approximately computing trace of an
%   inverse matrix when calculating gradient. Default 0 uses exact
%   calculation instead of sampling.
%   printStuff: verbosity
%   fixedIntercept: whether intercept should be fixed. 
%   Default 1. fixedIntercept=0 currently unsupported.
%   convergenceTol: terminates when objective function improves less than
%   this; when log-likelihood improves by less, declare convergence
%   maxReps: maximum number of steps to perform
%   minReps: starts checking for convergence after this number of steps
%   trustRegionSizeParam: hyperparameter used to tune the step size
%   maxiter: maximum number of attempts to adjust step size
%   trustRegionRhoLB: lower bound of the ratio of the actual and predicted
%   change of likelihood, i.e., a ratio lower than this threshold will lead
%   to shrinkage of the trust region
%   trustRegionRhoUB: upper bound of the ratio of the actual and predicted
%   change of likelihood, i.e., a ratio higher than this threshold will lead
%   to expansion of the trust region
%   trustRegionScalar: scaling factor for adjusting (shrink or expand) the
%   step size
%   deltaGradCheck: whether to check the norm of gradient from iter to iter
%   useTR: whether to use the trust-region algorithm to adjust for the step
%   size
%   refCol: which column of the annotaion matrix to use as the baseline
%   or refernece for computing enrichment
%   chisqThreshold: threshold at which large effect loci will be discarded,
%   together with their entire LD block. 
%
%   Output arguments:
%   estimate: struct containing the following fields: (to be updated)
%       params: estimated paramters
%       h2: heritability estimates for each annotation; equal to \sum_j
%       h^2(j) a_kj, where a_kj is annotation value for annotation k and
%       SNP j, and h^2(j) is the per-SNP h2 of SNP j
%       annotSum: sum of annotation values for each annotation across SNPs
%       logLikelihood: likelihood of sumstats at optimum
%       nn: sample size or 1/intercept if intercept was learned
%       paramsVar: approx sampling covariance of the parameters
%       SE: standard error of total h2 and *enrichment*
%       intercept: estimate of the intercept
%       interceptSE: standard error of the intercept
%
%   steps: struct containing the following fields:
%       reps: number of steps before stopping
%       params: parameter values at each step
%       obj: objective function value (-logLikelihood) at each step
%       gradient: gradient of the objective function at each step
%
%   r2_proxy: struct with one entry per LD block, describing the LD proxies
%   that were used for missing SNPs. Contains the following fields:
%       oldIndices: indices present in whichIndicesAnnot but not
%       whichIndicesSumstats
%       newIndices: indices present in whichIndicesSumstats to which
%       oldIndices were mapped
%       r2: squared "correlation" (although n.b. it can be >1) between
%       oldIndices and their LD proxies

%   diagnostics: struct containing the following fields
%       paramVar: model-based covariance matrix of the coefficients;
%       paramSandVar: sandwich covariance matrix of the coefficients;
%       paramJackVar: jackknife covariance matrix of the coefficeints;
%       cov: model-based covariance matrix of [total partitioned] heritability;
%       sand_cov: sandwich covariance matrix of [total partitioned] heritability;
%       jack_cov: jackknife covariance matrix of [total partitioned] heritability;


% initialize
p=inputParser;
if iscell(P)
    mm = sum(cellfun(@length,P));
else
    mm = length(P);
end

addRequired(p, 'Z', @(x)isvector(x) || iscell(x))
addRequired(p, 'P', @(x)ismatrix(x) || iscell(x))
addOptional(p, 'sampleSize', 1, @isscalar)
addOptional(p, 'whichIndicesSumstats', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'whichIndicesAnnot', cellfun(@(x){true(size(x))},Z), @(x)isvector(x) || iscell(x))
addOptional(p, 'annot', cellfun(@(x){true(size(x))},Z), @(x)size(x,1)==mm || iscell(x))
addOptional(p, 'linkFn', @(a,x)log(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'linkFnGrad', @(a,x)exp(a*x).*a./(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'params', [], @isvector)
addOptional(p, 'fixedIntercept', true, @isscalar)
addOptional(p, 'intercept', 1, @isscalar)
addOptional(p, 'printStuff', true, @isscalar)
addOptional(p, 'noSamples', 0, @(x)isscalar(x) & round(x)==x)
addOptional(p, 'convergenceTol', 1e-1, @isscalar)
addOptional(p, 'maxReps', 1e2, @isscalar)
addOptional(p, 'minReps', 3, @isscalar)
addOptional(p, 'trustRegionSizeParam', 1e-3, @isscalar)
addOptional(p, 'maxiter', 20, @isscalar)
addOptional(p, 'trustRegionRhoLB', 1e-4, @isscalar)
addOptional(p, 'trustRegionRhoUB', 0.99, @isscalar)
addOptional(p, 'trustRegionScalar', 10, @isscalar)
addOptional(p, 'deltaGradCheck', false, @isscalar)
addOptional(p, 'useTR', true, @isscalar)
addOptional(p, 'refCol', 1, @isnumeric)
addOptional(p, 'chisqThreshold', inf, @isscalar)

parse(p, Z, P, varargin{:});

% turns p.Results.x into just x
fields = fieldnames(p.Results);
for k=1:numel(fields)
    line = sprintf('%s = p.Results.%s;', fields{k}, fields{k});
    eval(line);
end

% throw out LD blocks with a chisq statistic greater than threshold
maxChisq = cellfun(@(x)max(x.^2), Z);
keep = maxChisq <= chisqThreshold;
if any(~keep)
    if printstuff
        fprintf('Discarding %d out of %d LD blocks due to chisq threshold\n', sum(~keep), length(Z))
    end
    Z = Z(keep);
    P = P(keep);
    whichIndicesAnnot = whichIndicesAnnot(keep);
    whichIndicesSumstats = whichIndicesSumstats(keep);
    annot = annot(keep);
end

blocksize = cellfun(@length, Z);
noAnnot = size(annot{1},2);
annotSum = cellfun(@(x){sum(x,1)},annot);
annotSum = sum(vertcat(annotSum{:}));
noBlocks = length(annot);
if isempty(params)
    params = zeros(noAnnot,1);
    if ~fixedIntercept
        params(end+1,1) = 1;
    end
end
smallNumber = 1e-6;

noParams = length(params) - 1 + fixedIntercept;
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
fixedIntercept = fixedIntercept;
intercept = intercept;
noSamples = noSamples;
allSteps=zeros(min(maxReps,1e6),noParams+1-fixedIntercept);
allValues=zeros(min(maxReps,1e6),1);
allGradients=allSteps;

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

% Objective function
if fixedIntercept
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params)}, annot),...
        P, sampleSize, whichIndicesAnnot, 1);

else
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params(1:end-1))}, annot),...
        P, sampleSize, whichIndicesAnnot, params(end));

end
newObjVal = objFn(params);

% main loop for updating parameter values
for rep=1:maxReps
    if printStuff
        disp(rep)
    end

    % Compute gradient and hessian by summing over blocks
    gradient = 0;
    hessian = 0;

    parfor block = 1:noBlocks
        % Effect-size variance for each sumstats SNPs, adding
        % across annotation  SNPs
        sigmasq = accumarray(whichSumstatsAnnot{block}, ...
            linkFn(annot{block}, params(1:noParams)));
        
        % Gradient of the effect-size variance for each sumstats SNP
        sg = linkFnGrad(annot{block}, params(1:noParams));
        sigmasqGrad = zeros(length(sigmasq), noParams);

        for kk = 1:noParams
            sigmasqGrad(:,kk) = accumarray(whichSumstatsAnnot{block}, sg(:,kk));
        end
        
        % Hessian of the log-likelihood
        hessian = hessian + ...
            GWASlikelihoodHessian(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, intercept, fixedIntercept)';
        
        % Gradient of the log-likelihood
        if noSamples > 0
            gradient = gradient + GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndicesSumstats{block}, samples{block})';
        else
            gradient = gradient + GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndicesSumstats{block}, intercept, fixedIntercept)';
        end
    end

    % Compute step (with the option to use trust region for adaptive step size)
    if ~useTR
        params = params - (hessian + trustRegionSizeParam * diag(diag(hessian)) + ...
            1e-2 * trustRegionSizeParam * mean(diag(hessian)) * eye(size(hessian))) \ gradient;

        % New objective function value
        newObjVal = objFn(params);

        if rep > 1 % update using heuristics
            while allValues(rep-1) - newObjVal < -smallNumber
                trustRegionSizeParam = 2*trustRegionSizeParam;
                warning('Objective function increased at iteration %d; increasing stepSizeParam to %.2f', rep, trustRegionSizeParam)
                params = params - (hessian + trustRegionSizeParam * diag(diag(hessian)) + ...
                    1e-2 * trustRegionSizeParam * mean(diag(hessian)) * eye(size(hessian))) \ gradient;
                newObjVal = objFn(params);
            end
        end
    else
        % initialize some fixed values for step size tuning;
        oldParams = params;
        trustRegionLam = trustRegionSizeParam;
        no_update = true;
        iter = 0;
        rho_arr = [];

        % define predicted change of likelihood
        delta_pred_ll = @(x)transpose(x)*gradient - 0.5*transpose(x)*(hessian*x);

        while no_update && iter < maxiter
            iter = iter + 1;

            % propose a new step (at the current value of trustRegionLam)
            hess_mod = hessian + trustRegionLam*diag(diag(hessian));

            % to prevent NaN step size, add scaled diag matrix to hess
            if rcond(hess_mod) < 1e-30
                hess_mod = hess_mod + 1e-2 * trustRegionSizeParam *...
                    mean(diag(hessian)) * eye(size(hessian));
            end

            stepSize = hess_mod \ gradient;

            if printStuff; disp('Newly proposed step size:'); end

            % Assess the proposed step -- eval likelihood (and gradient if
            % specified) at the new values of parameter
            params_propose = oldParams - stepSize;
            newObjVal_propose = objFn(params_propose);

            if deltaGradCheck
                gradient_propose = 0;
                for block = 1:noBlocks
                    sigmasq = linkFn(annot{block}, params_propose(1:noParams));
                    sigmasqGrad = linkFnGrad(annot{block}, params_propose(1:noParams)) .* annot{block};

                    if noSamples > 0
                        gradient_propose = gradient_propose + ...
                            GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, samples{block})';
                    else
                        gradient_propose = gradient_propose + ...
                            GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, ...
                            intercept, fixedIntercept)';
                    end
                end
            end

            % Evaluate the proposed step size
            delta_actual = newObjVal_propose - newObjVal;
            delta_pred = delta_pred_ll(stepSize);

            % Assess the quality of the step
            if delta_actual > 0
                if printStuff; disp('Neg-logL increased. Reject step size.'); end
                rho = -1;
            elseif deltaGradCheck
                if norm(gradient_propose) > (2*norm(gradient))
                    if printStuff; disp('Gradient changes too much. Reject step size.'); end
                    rho = -1;
                else
                    rho = abs(delta_actual / delta_pred);
                end
            else
                rho = abs(delta_actual / delta_pred);
            end

            % Accept or reject the update
            if rho > trustRegionRhoLB
                no_update = false; % stops the search of step size

                newObjVal = objFn(params_propose);
                params = params_propose;
                if printStuff
                    disp('Accepted step size')
                    disp('Updated objective function value:')
                    disp(newObjVal)
                end
            end

            % Update the trust region penalty
            if rho < trustRegionRhoLB
                if printStuff
                    disp(['Shrink the trust region / ' ...
                        'Increase the step size penalty term.'])
                end
                trustRegionLam = trustRegionLam * trustRegionScalar;
            elseif rho > trustRegionRhoUB
                if printStuff
                    disp(['Expand the trust region / ' ...
                        'Reduce the step size penalty term.'])
                end
                trustRegionLam = trustRegionLam / trustRegionScalar;
            end
            rho_arr(end+1) = rho;
        end
        if printStuff
            disp('History of the rho values:')
            disp(rho_arr)
        end
        % If adjustment is not successful, simply
        % remove the gradient check and move on
        if iter == maxiter
            % give warnings of the updating
            warning(['After attempts to adjust step size, ' ...
                'gradient still changes too much'])
            warning('Ignore the gradient check and move on')
            rho = abs(delta_actual / delta_pred);

            % Accept the change unless the ratio is too small
            if rho > trustRegionRhoLB
                if printStuff; disp('Accepted step size'); end
                no_update = false;
                newObjVal = objFn(params_propose);
                params = params_propose;
                if printStuff
                    disp('Updated parameter values:')
                    disp(params(1:5)')
                end
            end
        else
            if printStuff
                disp('Updated parameter values:')
                disp(params(1:5)')
            end
        end

        if ~fixedIntercept
            intercept = params(end);
        end

        allValues(rep)=newObjVal;
        allSteps(rep,:)=params;

        if rep > minReps
            if allValues(rep-minReps) - newObjVal < minReps * convergenceTol
                break;
            end
        end
    end
end



%% Post-maximization computation
% Compute block-specific gradient (once at the estimate)
grad_blocks = zeros(noBlocks, noParams + 0^fixedIntercept);
hess_blocks = zeros(noParams + 0^fixedIntercept, noParams + 0^fixedIntercept, noBlocks);
snpGrad = cell(noBlocks,1);

parfor block = 1:noBlocks
    sigmasq = accumarray(whichSumstatsAnnot{block}, ...
        linkFn(annot{block}, params(1:noParams)));

    sg = linkFnGrad(annot{block}, params(1:noParams));
    sigmasqGrad = zeros(length(sigmasq), noParams);

    for kk=1:noParams
        sigmasqGrad(:,kk) = accumarray(whichSumstatsAnnot{block}, sg(:,kk));
    end

    if noSamples > 0
        grad_blocks(block,:) = GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, samples{block})';
    else
        [grad_blocks(block,:), nodeGrad] = GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndicesSumstats{block}, intercept, fixedIntercept);
        snpGrad{block} = sg(:,1) .* nodeGrad(whichSumstatsAnnot{block});
    end

   hess_blocks(:,:,block) = GWASlikelihoodHessian(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndicesSumstats{block}, intercept, fixedIntercept)';
        
end
hessian = sum(hess_blocks,3);
grad = sum(grad_blocks,1);
psudojackknife = zeros(noBlocks, noParams + 0^fixedIntercept);
for block = 1:noBlocks
    psudojackknife(block,:) = params + ( hessian - hess_blocks(:,:,block) + 1e-12*eye(size(hessian))) \ (grad_blocks(block,:) - grad)';
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

% h2 estimates for each leave-one-out subset of the data
if nargout >= 5
    jackknife.params = psudojackknife;
    jackknife.h2 = zeros(size(psudojackknife));
    for block = 1:noBlocks
        for jk = 1:noBlocks
            perSNPh2_jk = linkFn(annot{block}, psudojackknife(jk, 1:noParams)');
            jackknife.h2(jk,:) = jackknife.h2(jk,:) + sum(perSNPh2_jk .* annot_unnormalized{block});
        end
    end
    jackknife.score = snpGrad;
end

estimate.params = params;
estimate.h2 = h2Est;
estimate.annotSum = annotSum;
estimate.logLikelihood = -newObjVal;

% Enrichment only calculated if first annotation is all-ones vector
if all(cellfun(@(a)all(a(:,1)==1),annot_unnormalized))
    estimate.enrichment = (h2Est./annotSum) / (h2Est(1)/annotSum(1));
end

%% Compute covariance of the parameter estimates
% Approximate fisher information (AI)
FI = -hessian;

if any(diag(FI)==0)
    warning(['Some parameters have zero fisher information. ' ...
        'Regularizing FI matrix to obtain standard errors.'])
    FI = FI + smallNumber*eye(size(FI));
end
naiveVar = pinv(FI);

% Approximate empirical covariance of the score (using the block-wise estimates)
empVarApprox = cov(grad_blocks) * noBlocks;
sandVar = naiveVar*(empVarApprox*naiveVar);
jkVar = cov(psudojackknife) * (noBlocks-2);

% Turn annotation and coefficients to genetic variances
annot_mat = vertcat(annot{:});
link_val = linkFn(annot_mat, params(1:noParams));
link_jacob = linkFnGrad(annot_mat, params(1:noParams)) .* annot_mat;
G = sum(link_val);
J = sum(link_jacob, 1);

%% Variance of enrichment via chain rule
dMdtau = 1/(G^2)*(G*link_jacob - kron(link_val, J));
dMdtau_A = transpose(dMdtau) * annot_mat;
p_annot = mean(annot_mat, 1)';

if ~fixedIntercept
    estimate.intercept = params(end);
    estimate.interceptSE = sqrt(naiveVar(end,end));
    estimate.interceptSandSE = sqrt(sandVar(end,end));
    naiveVar = naiveVar(1:noParams, 1:noParams);
    sandVar = sandVar(1:noParams, 1:noParams);
    jkVar = jkVar(1:noParams,1:noParams);
else
    estimate.intercept = intercept;
    estimate.interceptSE = 0;
    estimate.interceptSandSE = 0;
end

% naive SE estimator
SE_prop_h2 = sqrt(diag(transpose(dMdtau_A)*(naiveVar*dMdtau_A)));
enrich_SE = SE_prop_h2(1:noParams) ./ p_annot(1:noParams);
enrich_SE(1) = sqrt(J*(naiveVar*transpose(J)));

% robust / Huber-White estimator
sandSE_prop_h2 = sqrt(diag(transpose(dMdtau_A)*(sandVar*dMdtau_A)));
enrich_sandSE = sandSE_prop_h2(1:noParams) ./ p_annot(1:noParams);
enrich_sandSE(1) = sqrt(J*(sandVar*transpose(J)));

% jackknife estimator
jkSE_prop_h2 = sqrt(diag(transpose(dMdtau_A)*(jkVar*dMdtau_A)));
enrich_jkSE = jkSE_prop_h2(1:noParams) ./ p_annot(1:noParams);
enrich_jkSE(1) = sqrt(J*(jkVar*transpose(J)));

%% Variance of annot-h2 via chain rule
dMdtau_J = transpose(link_jacob) * annot_mat;

% naive SE estimator
naive_cov = transpose(dMdtau_J)*(naiveVar*dMdtau_J);
SE_h2 = sqrt(diag(naive_cov));

% robust / Huber-White estimator
sand_cov = transpose(dMdtau_J)*(sandVar*dMdtau_J);
sandSE_h2 = sqrt(diag(sand_cov));

% jackknife estimator
jk_cov = transpose(dMdtau_J)*(jkVar*dMdtau_J);
jkSE_h2 = sqrt(diag(jk_cov));


%% Compute p-values for enrichment (defined as difference)
% based on naive SE
naive_pval = enrichment_pval(estimate.h2, SE_h2, naive_cov, p_annot', 'refCol', refCol);

% based on robust SE
sand_pval = enrichment_pval(estimate.h2, sandSE_h2, sand_cov, p_annot', 'refCol', refCol);

% based on jk SE
jk_pval = enrichment_pval(estimate.h2, jkSE_h2, jk_cov, p_annot', 'refCol', refCol);

%% Record estimates
estimate.paramSE = sqrt(diag(naiveVar));
estimate.paramSandSE = sqrt(diag(sandVar));
estimate.paramJackSE = sqrt(diag(jkVar));
estimate.p_annot = p_annot';
estimate.SE = enrich_SE';
estimate.sandSE = enrich_sandSE';
estimate.jkSE = enrich_jkSE';
estimate.h2SE = SE_h2';
estimate.h2sandSE = sandSE_h2';
estimate.h2jkSE = jkSE_h2';
estimate.enrichPval = naive_pval;
estimate.enrichsandPval = sand_pval;
estimate.enrichjkPval = jk_pval;
estimate.coefPval = compute_pval(params, naiveVar);
estimate.coefsandPval = compute_pval(params, sandVar);
estimate.coefjkPval = compute_pval(params, jkVar);

%% Record variance-covariance matrices
diagnostics.paramVar = naiveVar;
diagnostics.paramSandVar = sandVar;
diagnostics.paramJackVar = jkVar;
diagnostics.cov = naive_cov;
diagnostics.sand_cov = sand_cov;
diagnostics.jack_cov = jk_cov;

end


function [estimate, steps] = h2newton_testing(Z,P,varargin)
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
%   trustRegionSizeParam: hyperparameter used to tune the step size
%   useTR: whether to use the trust-region algorithm to adjust for the step
%   size
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
addOptional(p, 'linkFn', @(a,x)log(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'linkFnGrad', @(a,x)exp(a*x).*a./(1+exp(a*x))/mm, @(f)isa(f,'function_handle'))
addOptional(p, 'params', [], @isvector)
addOptional(p, 'fixedIntercept', true, @isscalar)
addOptional(p, 'printStuff', true, @isscalar)
addOptional(p, 'noSamples', 0, @(x)isscalar(x) & round(x)==x)
addOptional(p, 'convergenceTol', 1e-1, @isscalar)
addOptional(p, 'maxReps', 1e2, @isscalar)
addOptional(p, 'minReps', 3, @isscalar)
addOptional(p, 'trustRegionSizeParam', 1e-3, @isscalar)
addOptional(p, 'deltaGradCheck', false, @isscalar)
addOptional(p, 'useTR', true, @isscalar)

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
        params(end+1,1) = 1;
    end
end
smallNumber = 1e-6;
noParams = length(params) - 1 + fixedIntercept;

annot_unnormalized = annot;
annot = annot;
% annot = cellfun(@(a){mm*a./max(1,annotSum)},annot);

if fixedIntercept
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params)}, annot),...
        P, sampleSize, whichIndices, 1);

else
    objFn = @(params)-GWASlikelihood(Z,...
        cellfun(@(x){linkFn(x, params(1:end-1))}, annot),...
        P, sampleSize, whichIndices, params(end));

end
intercept = 1;

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
    parfor block = 1:noBlocks
        sigmasq = linkFn(annot{block}, params(1:noParams));
        sigmasqGrad = linkFnGrad(annot{block}, params(1:noParams));

        hessian = hessian + ...
            GWASlikelihoodHessian(Z{block},sigmasq,P{block},...
            sampleSize, sigmasqGrad, whichIndices{block}, intercept, fixedIntercept)';

        if noSamples > 0
            gradient = gradient + ...
                GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndices{block}, samples{block})';
        else
            gradient = gradient + ...
                GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                sampleSize, sigmasqGrad, whichIndices{block}, intercept, fixedIntercept)';
        end
    end

    % Compute step (with the option to use trust region for adaptive step size)
    if ~useTR
        params = params - (hessian + trustRegionSizeParam * diag(diag(hessian)) + ...
            1e-2 * trustRegionSizeParam * mean(diag(hessian)) * eye(size(hessian))) \ gradient;
    
        % New objective function value
        newObjVal = objFn(params);
    
        if rep > 1
            while allValues(rep-1) - newObjVal < -smallNumber
                trustRegionSizeParam = 2*trustRegionSizeParam;
                warning('Objective function increased at iteration %d; increasing stepSizeParam to %.2f', rep, trustRegionSizeParam)
                params = params - (hessian + trustRegionSizeParam * diag(diag(hessian)) + ...
                    1e-2 * trustRegionSizeParam * mean(diag(hessian)) * eye(size(hessian))) \ gradient;
                newObjVal = objFn(params);
            end
        end
    else
        % initialize fixed values for step size tuning; set hyperparameters
        oldParams = params;
        trustRegion_lam = trustRegionSizeParam;
        no_update = true; 
        maxiter = 50; iter = 0;
        rho_arr = [];
        eta1 = 0.0001; eta2 = 0.5; 

        % define predicted change of likelihood
        delta_pred_ll = @(x)transpose(x)*gradient - 0.5*transpose(x)*(hessian*x);

        while no_update && iter < maxiter
            iter = iter + 1;
        
            % propose a new step (at the current value of trustRegion_lam)
            hess_mod = hessian + trustRegion_lam*diag(diag(hessian));

            % to prevent NaN step size, add scaled diag matrix to hess
            if rcond(hess_mod) < 1e-30 % prevent NaN step size
                hess_mod = hess_mod + 1e-2 * trustRegionSizeParam * mean(diag(hessian)) * eye(size(hessian));
            end

            step = hess_mod \ gradient;
            disp('Newly proposed step size:')
            disp(step(1:min(3,length(step)))')

            % Assess the proposed step -- eval likelihood and gradient at
            % the new parameter estimate
            params_propose = oldParams - step;
            newObjVal_propose = objFn(params_propose);
            gradient_propose = 0;
            
            if deltaGradCheck
                parfor block = 1:noBlocks
                    sigmasq = linkFn(annot{block}, params_propose(1:noParams));
                    sigmasqGrad = linkFnGrad(annot{block}, params_propose(1:noParams));

                    if noSamples > 0
                        gradient_propose = gradient_propose + ...
                            GWASlikelihoodGradientApproximate(Z{block},sigmasq,P{block},...
                            sampleSize, sigmasqGrad, whichIndices{block}, samples{block})';
                    else
                        gradient_propose = gradient_propose + ...
                            GWASlikelihoodGradient(Z{block},sigmasq,P{block},...
                            sampleSize, sigmasqGrad, whichIndices{block}, intercept, fixedIntercept)';
                    end
                end
            end
            
            % Evaluate the proposed step size
            delta_actual = newObjVal_propose - newObjVal;
            delta_pred = delta_pred_ll(step);
        
            % Assess the quality of the step
            if delta_actual > 0
                disp('Neg-logL increased. Reject step size.')
                rho = -1;
            elseif deltaGradCheck
                if norm(gradient_propose) > (2*norm(gradient))
                    disp('Gradient changes too much. Reject step size.')
                    rho = -1;
                else
                    rho = abs(delta_actual / delta_pred);
                end
            else
                rho = abs(delta_actual / delta_pred);
            end
        
            % Accept or reject the update
            if rho > eta1
                disp('Accepted step size')
                no_update = false; % stops the search of step size

                newObjVal = objFn(params_propose);
                params = params_propose;
            end
        
            % Update the trust region penalty (requires hyperparameters)
            if rho < eta1
                disp(['Shrink the trust region / ' ...
                    'Increase the step size penalty term.'])
                trustRegion_lam = trustRegion_lam*10;
            elseif rho > eta2
                disp(['Expand the trust region / ' ...
                    'Reduce the step size penalty term.'])
                trustRegion_lam = trustRegion_lam/2;        
            end
            rho_arr(end+1) = rho;
        end
        disp('History of the rho values:')
        disp(rho_arr)

        % If adjustment is not successful, simply 
        % remove the gradient check and move on
        if iter == maxiter
            % give warnings of the updating
            if norm(gradient_propose) > (2*norm(gradient))
                warning(['After attempts to adjust step size, ' ...
                    'gradient still changes too much'])
            elseif delta_actual > 0
                warning(['After attempts to adjust step size, ' ...
                    'neg-logL still increases'])
            end
            warning('Ignore the gradient check and move on')

            rho = abs(delta_actual / delta_pred);
        
            % Accept the change unless the ratio is too small
            if rho > eta1
                disp('Accepted step size')
                no_update = false;
                newObjVal = objFn(params_propose);
                params = params_propose;
                disp('Updated parameter values:')
                disp(params(1:min(5,length(params)))')
            end
        else
            disp('Updated parameter values:')
            disp(params(1:min(5,length(params)))')
        end
    end

    if ~fixedIntercept
        intercept = params(end);
    end

    allValues(rep)=newObjVal;
    allGradients(rep,:) = gradient;
    allSteps(rep,:)=params;

    if rep > minReps
        if allValues(rep-minReps) - newObjVal < minReps * convergenceTol
            break;
        end
    end

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
params_inv = pinv(FI);

% Turn annotation and coefficients to genetic variances
annot_mat = vertcat(annot{:});
link_val = linkFn(annot_mat, params(1:noParams));
link_jacob = linkFnGrad(annot_mat, params(1:noParams));
G = sum(link_val);
J = sum(link_jacob, 1);

% Variance of h2 estimates via chain rule
dMdtau = 1/(G^2)*(G*link_jacob - kron(link_val, J));
dMdtau_A = transpose(dMdtau) * annot_mat;
p_annot = mean(annot_mat, 1)';
SE_h2 = sqrt(diag(transpose(dMdtau_A)*(params_inv*dMdtau_A)));
enrich_SE = SE_h2(1:noParams) ./ p_annot(1:noParams);
enrich_SE(1) = sqrt(J*(params_inv*transpose(J)));

% Record variance and SE
estimate.paramVar = params_inv;
estimate.SE = enrich_SE';

if ~fixedIntercept
    estimate.intercept = params(end);
    estimate.interceptSE = sqrt(estimate.paramsVar(end,end));
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


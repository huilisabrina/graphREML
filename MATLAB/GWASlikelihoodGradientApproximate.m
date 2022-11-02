function [grad] = GWASlikelihoodGradientApproximate(Z, sigmasq, P, nn, delSigmaDelA, whichSNPs, samples)
% GWASlikelihoodGradient computes gradient of the likelihood of the GWAS 
% sumstats alphaHat under a gaussian model:
%                   beta ~ MVN(mu,diag(sigmasq))
%                   alphaHat|beta ~ MVN(R*beta, R/nn)
%                   inv(R) = P.
% 
% The GWAS SNPs in alphahat should be a subset of those in P, and in the
% same order; boolean vector whichSNPs should be true for rows/columns of P
% corresponding to one of these SNPs, false elsewhere.
%
% sigmasq should be the same size as alphahat, such that missing
% SNPs are modeled as having zero effect-size variance.
%
% Argument delSigmaDelA is the gradient of sigmasq w.r.t. some parameters A.
% GWASlikelihoodGradient computes the gradient of the likelihood w.r.t.
% A, i.e.:
% grad_A loglikelihood = delSigmaDelA * grad_sigmasq loglikelihood 
%
% Units of alphaHat should be either (1) normalized effect size estimates,
% ie sample correlations, or (2) Z scores. In case (1), nn should be the
% number of individuals in the GWAS. In case (2), nn should be 1, and the
% vector of sigmasq values should be multiplied by nn.
% Optionally, arguments alphaHat, sigmasq, P, whichSNPs can be specified as
% cell arrays with one cell for each LD block.
%
% Optionally, arguments alphaHat, sigmasq, P, whichSNPs can be specified as
% cell arrays with one cell for each LD block.
%
% Optionally, if it is desired to specify the mean of beta, ie
%                   beta ~ MVN(mu, diag(sigmasq))
% call GWASlikelihoodGradient( alphahat - P \ mu, ...)
%
% Optionally, if it is desired to model an unknown amount of population
% stratification/relatedness, or if the sample size is unknown, specify
% fixedIntercept = true. This modifies the model as:
% alphaHat|beta ~ MVN(R*beta, R * (1/nn + a))
% where a==0, and it computes the derivative of the log-likelihood with 
% respect to a; this will be the last element of grad.


if iscell(P) % handle cell-array-valued inputs
    if nargin < 5
        whichSNPs = cellfun(@(x)true(size(x),Z),'UniformOutput', false);
    end
    assert(iscell(Z) & iscell(whichSNPs) & iscell(sigmasq))
    grad = cellfun(@(a,s,p,dS,w,x)GWASlikelihoodGradient(a,s,p,nn,dS,w,x),...
        Z,sigmasq,P,delSigmaDelA,whichSNPs,samples, 'UniformOutput', false);
else
    % unit conversion
    Z = Z/sqrt(nn);
    
    % handle missing rows/cols of P
    if ~islogical(whichSNPs)
        whichSNPs = unfind(whichSNPs,length(P));
    end
    pnz = diag(P)~=0;
    assert(all(pnz(whichSNPs)))
    mm = sum(pnz);
    P = P(pnz,pnz);
    whichSNPs = whichSNPs(pnz);
    
    % M = Sigma + P/nn is the covariance matrix of betaHat = P/P11 * alphaHat
    M = zeros(mm,1);
    M(whichSNPs) = sigmasq;
    M = P/nn + speye(mm).*M;

    
    
    % betahat = P/P11 * alphaHat
    betahat = precisionMultiply(P, Z, whichSNPs);
    b = precisionDivide(M, betahat, whichSNPs);
    
    % quadratic term
    delQuadratic = -sum(delSigmaDelA .* b.^2);

    % trace term
    yy = precisionDivide(M, samples,whichSNPs);
    delLogDetP = delQuadratic;
    for ii = 1:size(delSigmaDelA,2)
        % approximately Tr(inv(M) * diag(delSigmaDelA(:,ii)))
        delLogDetP(ii) = sum((delSigmaDelA(:,ii) .* samples) .* yy, 'all');
    end
    
    grad = - 1/2 * (delLogDetP + delQuadratic);
    

end
end


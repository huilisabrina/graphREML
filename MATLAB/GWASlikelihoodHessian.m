function [J] = GWASlikelihoodHessian(Z, sigmasq, P, nn, delSigmaDelA, whichSNPs, fixedIntercept)
% GWASlikelihoodHessian computes the Hessian of the likelihood of the GWAS 
% sumstats alphaHat under a gaussian model:
%                   beta ~ MVN(0,diag(sigmasq))
%                   Z|beta ~ MVN(sqrt(n)*R*beta, R)
%                   inv(R) = P.
% 
% The GWAS SNPs in Z should be a subset of those in P, and in the
% same order; boolean vector whichSNPs should be true for rows/columns of P
% corresponding to one of these SNPs, false elsewhere (or, specify indices)
%
% sigmasq should be the same size as Z, such that missing
% SNPs are modeled as having zero effect-size variance.
%
% Argument delSigmaDelA is the gradient of sigmasq w.r.t. some parameters A.
%
% Optionally, arguments Z, sigmasq, P, whichSNPs can be specified as
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

if nargin < 7
    fixedIntercept = true;
elseif ~fixedIntercept
    error('Not yet supported')
end

if iscell(P) % handle cell-array-valued inputs
    if nargin < 6
        whichSNPs = cellfun(@(x)true(size(x),Z),'UniformOutput', false);
    end
    assert(iscell(Z) & iscell(whichSNPs) & iscell(sigmasq))
    J = cellfun(@(a,s,p,dS,w)GWASlikelihoodHessian(a,s,p,nn,dS,w,fixedIntercept),...
        Z,sigmasq,P,delSigmaDelA,whichSNPs, 'UniformOutput', false);
else
    
    % handle missing rows/cols of P
    if ~islogical(whichSNPs)
        whichSNPs = unfind(whichSNPs,length(P));
    end
    pnz = diag(P)~=0;
    assert(all(pnz(whichSNPs)))
    mm = sum(pnz);
    P = P(pnz,pnz);
    whichSNPs = whichSNPs(pnz);
    
    % M = nn*Sigma + P is the covariance matrix of P*z
    M = zeros(mm,1);
    M(whichSNPs) = sigmasq;
    M = P + nn * speye(mm).*M;
    
    % betahat = P/P11 * Z
    betahat = precisionMultiply(P, Z, whichSNPs);
    b = precisionDivide(M, betahat, whichSNPs);
    
    % approximate hessian
    b_scaled = b .* delSigmaDelA;
    J = -1/2 * nn^2 * b_scaled' * precisionDivide(M, b_scaled, whichSNPs);
   
end
end


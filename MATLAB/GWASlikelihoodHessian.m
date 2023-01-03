function [J] = GWASlikelihoodHessian(Z, sigmasq, P, nn, delSigmaDelA, whichSNPs, intercept, fixedIntercept)
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
% respect to a; this will be the last element of J. 

if nargin < 7
    intercept = 1;
end
assert(isscalar(intercept) && all(intercept>=0,'all'))
if nargin < 8
    fixedIntercept = true;
end

if iscell(P) % handle cell-array-valued inputs
    if nargin < 6
        whichSNPs = cellfun(@(x)true(size(x),Z),'UniformOutput', false);
    end
    assert(iscell(Z) & iscell(whichSNPs) & iscell(sigmasq))
    J = cellfun(@(a,s,p,dS,w)GWASlikelihoodHessian(a,s,p,nn,dS,w,intercept,fixedIntercept),...
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
    
    % M == E(xx')
    M = sparse(find(whichSNPs), find(whichSNPs), nn*sigmasq, mm, mm);
    M = M + intercept * P;
    
    % betahat = P/P11 * Z
    x = precisionMultiply(P, Z, whichSNPs);

    % M * betaHat
    b = precisionDivide(M, x, whichSNPs);
    
    % derivative of M wrt parameters times b
    b_scaled = nn * b .* delSigmaDelA;
    
    % derivative of M wrt intercept times b
    if ~fixedIntercept
        b_scaled(:, end+1) = precisionMultiply(P, b, whichSNPs);
    end

    % approximate Hessian
    J = -1/2 * b_scaled' * precisionDivide(M, b_scaled, whichSNPs);
    
    % missing SNPs extra term
    if ~fixedIntercept
        mm0 = sum(~whichSNPs);
        J(end,end) = J(end,end) - 1/2 * mm0 * -1/intercept^2;
    end
   
end
end


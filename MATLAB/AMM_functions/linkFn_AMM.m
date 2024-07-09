function [sigmasq] = linkFn_AMM(annot,theta,nbaseline,Nsnp)
    % annot = annot matrix that includes both the baseline annot and the AMM annot
    % theta = vector of parameters
    % nbaseline = number of baseline annotations
    % Nsnp = total number of SNPs 

    % Note: the annotation matrix should have baseline annot BEFORE the AMM annotations

    % Generate indices to separate out baseline annot from AMM annot
    idxbaseline = logical([ones(1,nbaseline), zeros(1, (size(theta,1)-nbaseline))]');
    idxAMM = logical([zeros(1,nbaseline), ones(1, (size(theta,1)-nbaseline))]');

    % Baseline component of the link with overflow safeguard (both x and f_baseline are of length p)
    f_baseline = g(annot(:, idxbaseline) * theta(idxbaseline));

    % Excess h2 due to the kth nearest genes (both y and f_AMM are of length K)
    f_AMM = g(theta(idxAMM));

    sigmasq = (1/Nsnp) .* f_baseline .* (1 + annot(:, idxAMM) * f_AMM);

    % % util functions
    function y = g(x)
        y = x + log(1 + exp(-x));
        y(x < 0) = log(1 + exp(x(x < 0)));
    end

    % Normalize annot value after grouping (numannotvec)
    % sigmasq = (1/Nsnp) .* f_baseline .* (1 + annot(:, idxAMM) * (numannotvec(idxAMM) .* f_AMM));
end

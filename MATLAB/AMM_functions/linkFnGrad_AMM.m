function [sigmasqGrad] = linkFnGrad_AMM(annot,theta,nbaseline,Nsnp)
    % annot = annot matrix that includes both the baseline annot and the AMM annot
    % theta = vector of parameters
    % nbaseline = number of baseline annotations
    % Nsnp = total number of SNPs 

    % Note: the annotation matrix should have baseline annot BEFORE the AMM annotations


    % Generate indices to separate out baseline annot from AMM annot
    idxbaseline = logical([ones(1,nbaseline), zeros(1, (size(theta,1)-nbaseline))]');
    idxAMM = logical([zeros(1,nbaseline), ones(1, (size(theta,1)-nbaseline))]');

    % Baseline component of the link with overflow safeguard (both x and g_baseline are of length p)
    x = annot(:, idxbaseline) * theta(idxbaseline);
    g_baseline = g(x);
    % disp(size(g_baseline))

    % Excess h2 due to the kth nearest genes (both y and g_AMM are of length K; g_excess is of length p)
    g_AMM = g(theta(idxAMM));
    g_excess = (1 + annot(:, idxAMM) * g_AMM);
    % disp(size(g_excess))

    % Set up the variable (p x (A+K))
    sigmasqGrad = zeros(size(annot,1), size(theta,1));

    % linkFnGrad applied to the baseline
    % grad_fPb is p x K dimension
    h_baseline = h_base(annot(:, idxbaseline), x);
    grad_fPb = h_baseline / Nsnp;
    sigmasqGrad(:, idxbaseline) = g_excess .* grad_fPb;
    % disp(size(grad_fPb))

    % linkFnGrad applied to the AMM params
    grad_fP = h_amm(theta(idxAMM)) / Nsnp;
    sigmasqGrad(:, idxAMM) = g_baseline .* annot(:, idxAMM) .* grad_fP';
    % disp(size(grad_fP))

    % % util functions
    function y = g(x)
        y = x + log(1 + exp(-x));
        y(x < 0) = log(1 + exp(x(x < 0)));
    end

    function y = h_base(a,x)
        y = a ./ (1 + exp(-x));
        y(x < 0, :) = a(x < 0, :) .* exp(x(x < 0)) ./ (1 + exp(x(x < 0)));
    end

    function y = h_amm(x)
        y = 1 ./ (1 + exp(-x));
        y(x < 0) = exp(x(x < 0)) ./ (1 + exp(x(x < 0)));
    end

end
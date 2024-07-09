function [linkFn, linkFnGrad] = gen_linkFns(noSNPs)
    
    linkFn = @(annot, theta) (annot*theta + log(1 + exp(-annot*theta))) / noSNPs;
    linkFnGrad = @(a, t)g(a, t, noSNPs);

    function y = g(annot, theta, noSNPs)
        x = annot * theta;
        y = annot ./ noSNPs ./ (1 + exp(-x));
        y(x < 0, :) = annot(x < 0, :) .* exp(x(x < 0)) ./ (1 + exp(x(x < 0))) / noSNPs;
    end
end

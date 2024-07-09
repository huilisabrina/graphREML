function [y] = softmax_robust(x)
    y = x + log(1 + exp(-x));
    y(x < 0) = log(1 + exp(x(x < 0)));
end

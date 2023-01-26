function [l2] = calculate_LDscores(P,annot,whichIndices)
%calculate_LDscores compute stratified LD scores from precision matrices and
%annotation matrices
% Input arguments:
% P: cell array of precision matrices
% annot: cell array of annotation matrices
% whichIndices: indices of P{:} corresponding to rows of annot{:}
% Output arguments:
% l2: cell array of stratified LD scores

noAnnot = size(annot{1},2);
l2 = cell(size(P));
parfor b=1:length(P)
    indicesNotMissingFromP = find(any(P{b}));
    whichIndicesAnnot = lift(whichIndices{b},indicesNotMissingFromP);
    R = inv(P{b}(indicesNotMissingFromP,indicesNotMissingFromP));
    R = R(whichIndicesAnnot,whichIndicesAnnot);
    l2{b} = zeros(size(annot{b}));
    for k = 1:noAnnot
        l2{b}(:,k) = sum(R.^2.*annot{b}(:,k));
    end
end
end


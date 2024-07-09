function [aout] = makeAMMcols(a,vecin,nbaseline,namm)
% a = annot matrix including baseline annots and knearest annots matrix
% vecin = vector in AMM format (see Weiner github) that corresponds to how
% to bin annotations (not including baseline annots)
%
% nbaseline = number of baseline annotations
% namm = Number of AMM annotations
% colnamesout = column names out

if isempty(namm)
    sum_max_results = zeros(size(a,1),size(vecin,2));
    counter = 1;
    left_bound = nbaseline+1;
    for i = vecin
        right_bound = left_bound+i-1;
        sum_max_results(:,counter) = sum(a(:,left_bound:right_bound),2);
        left_bound = right_bound+1;
        counter = counter+1;
    end
    aout = horzcat(a(:,1:nbaseline),sum_max_results);

elseif isempty(nbaseline)
    sum_max_results = zeros(size(a,1),size(vecin,2));
    counter = 1;
    left_bound = 1;
    for i = vecin
        right_bound = left_bound+i-1;
        sum_max_results(:,counter) = sum(a(:,left_bound:right_bound),2);
        left_bound = right_bound+1;
        counter = counter+1;
    end
    aout = horzcat(sum_max_results,a(:,(namm+1):size(a,2))) ;
end

end

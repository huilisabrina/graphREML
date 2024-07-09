function [colnamesvecout] = makeAMMnames(vecin,nbaseline,namm)
    % vecin = vector in AMM format (see Weiner et al. AMM github) that corresponds to how to bin annotations (not including baseline annots)
    % nbaseline = number of baseline annotations
    % namm = Number of AMM annotations

    counter = 1;
    colnamesvecout = cell(size(vecin,1),1);

    if isempty(namm)
        left_bound = nbaseline+1;
        for i = vecin
            right_bound = left_bound+i-1;
            if right_bound == left_bound
                colnamesvecout{counter} = ['knearest',num2str(left_bound-nbaseline)];
            else
                colnamesvecout{counter} = ['knearest',num2str(left_bound-nbaseline),'to', ...
                    num2str(right_bound-nbaseline)];
            end
            left_bound = right_bound+1;
            counter = counter+1;
        end

    elseif isempty(nbaseline)
        left_bound = 1;
        for i = vecin
            right_bound = left_bound+i-1;
            if right_bound == left_bound
                colnamesvecout{counter} = ['knearest',num2str(left_bound)];
            else
                colnamesvecout{counter} = ['knearest',num2str(left_bound),'to', ...
                    num2str(right_bound)];
            end
            left_bound = right_bound+1;
            counter = counter+1;
        end
    end
end
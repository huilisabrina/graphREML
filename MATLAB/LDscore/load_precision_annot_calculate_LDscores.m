% 1/26/23 load precision matrices for a whole chromosome ,
% normalize them so the inverse has unit diagonal, 
% merge with baselineLD annotations, compute LD
% scores for selected annotations including null annotations.
% Save to files.

clear;clc

whichChromosomes = 22;
population = 'EUR';

addpath(genpath('C:/Users/huili/OneDrive/Documents/GitHub/h2-ldgm/MATLAB'));
addpath 'C:/Users/huili/OneDrive/Documents/GitHub/ldgm/MATLAB'
addpath 'C:/Users/huili/OneDrive/Documents/GitHub/ldgm/MATLAB/utility'
addpath 'C:/Users/huili/OneDrive/Documents/GitHub/ldgm/MATLAB/precision'
addpath 'C:/Users/huili/OneDrive/Documents/GitHub/h2-ldgm/sparseinv'

% Expects subdirectories called 'ldgms' containing precision matrices 
% and `ldscore` containing the .annot files, respectively
datapath = 'C:/Users/huili/OneDrive/Documents/GitHub/ldgm/data/';

%% 

P = []; snplists = []; AF = []; whichIndicesAnnot = []; mergedAnnot = []; 
for chromosome = whichChromosomes
    ldgms_dir = [datapath, 'ldgms/1kg_chr',num2str(chromosome),'_'];
    annot_dir = [datapath, ...
        'ldscore/baselineLD.',...
        num2str(chromosome),'.annot'];

    % BaselineLD annotations
    annotChr = readtable(annot_dir,'FileType','text');

    % Precision matrices
    [PChr, snplistsChr, AFChr] = loadLDGMs(ldgms_dir,'popnNames', population,...
        'normalizePrecision',true);
    P = [P; PChr]; %#ok<*AGROW> 
    snplists = [snplists; snplistsChr];
    AF = [AF; AFChr];

    % Merge annotations + precision matrices
    [whichIndicesAnnotChr, mergedAnnotChr] = mergesnplists(snplistsChr,annotChr,PChr);
    whichIndicesAnnot = [whichIndicesAnnot; whichIndicesAnnotChr];
    mergedAnnot = [mergedAnnot; mergedAnnotChr];
end


noSNPs = cellfun(@length, whichIndicesAnnot);
noSNPsTotal = cellfun(@length,P);
[noBlocks, noPops] = size(P);
clear *Chr


%% 

% Write normalized precision matrices to new subdirectory of datapath
mkdir([datapath, 'ldgm_normalized'])
counter = 0;
for chromosome = whichChromosomes
    snplist_dir = [datapath, 'ldgm/1kg_chr',num2str(chromosome),'_'];
    snplist_files = dir([ldgms_dir,'*','.snplist']);
    for block = 1:length(snplist_files)
        counter = counter + 1;
        blockname = snplist_files(block).name;
        blockname = blockname(1:end-length('.snplist'));
        writeedgelist([datapath, 'ldgm_normalized/', blockname, '.edgelist'],...
            P{counter})
    end
end

%% 

% Select which annotations to retain
whichAnnot = [5:9, 14:16, 19:20, 40:43];
annotNames = mergedAnnot{1}.Properties.VariableNames(whichAnnot)';

% Convert annotation tables into matrices
annotMatrices = cellfun(@(T)table2array(T(:,whichAnnot)), mergedAnnot,'UniformOutput',false);

% Add thin null annotations (random SNPs)
nullAnnotationFraction = [0.1 0.01];
for block = 1:noBlocks
    annotMatrices{block} = [annotMatrices{block}, ...
        rand(noSNPs(block),length(nullAnnotationFraction)) < nullAnnotationFraction];
end
for ii = 1:length(nullAnnotationFraction)
    annotNames = [annotNames; {sprintf('Random_snps_%dpct',round(100*nullAnnotationFraction(ii)))}];
end


% Add chunky null annotations (random blocks)
for ii = 1:length(nullAnnotationFraction)
    annotBlocks = randsample(1:noBlocks,ceil(noBlocks * nullAnnotationFraction(ii)),false);

    for block = 1:noBlocks
        annotMatrices{block} = [annotMatrices{block}, ...
            ones(noSNPs(block),1) * ismember(block, annotBlocks)];
    end
    annotNames = [annotNames; {sprintf('Random_blocks_%dpct',round(100*nullAnnotationFraction(ii)))}];
end

disp(annotNames)

%% 
tic
% LD scores, as a cell array of matrices
l2 = calculate_LDscores(P,annotMatrices,whichIndicesAnnot);
toc

%% 

% Save the annotation matrix and the LD scores block by block
mkdir([datapath, 'ldscore_block'])
for block = 1:length(snplist_files)
    disp(block)
    annot_snp_info = mergedAnnot{block}(:,1:4);

    blockname = snplist_files(block).name;
    blockname = blockname(1:end-length('.snplist'));

    % save the annotation information to files
    annot_fp = strjoin({datapath, 'ldscore_block/', blockname, '.annot'}, "");
    annot_score = array2table(annotMatrices{block}, 'VariableNames', annotNames);
    annot_out = horzcat(annot_snp_info, annot_score); 
    writetable(annot_out, annot_fp, ...
        'FileType', 'text', 'delimiter', '\t')

    % save the stratified LD scores to file
    annot_snp_info(:,4) = [];
    annot_snp_info = annot_snp_info(:, [1 3 2]);

    l2scoreNames = strcat(annotNames, 'L2');
    ldscore = array2table(l2{block}, 'VariableNames', l2scoreNames); 
    ldscore_out = horzcat(annot_snp_info, ldscore);
    ldscore_fp = strjoin({datapath, 'ldscore_block/', blockname, '.l2.ldscore'}, "");
    writetable(ldscore_out, ldscore_fp, ...
        'FileType', 'text', 'delimiter', '\t')

    % save baseline L2 score as the weights
    weights_ldscore = ldscore_out(:,1:4);
    weights_ldscore.Properties.VariableNames = ["CHR", "SNP", "BP", "L2"];
    weights_fp = strjoin({datapath, 'ldscore_block/', blockname, '_w.l2.ldscore'}, "");
    writetable(weights_ldscore, weights_fp, ...
        'FileType', 'text', 'delimiter', '\t')
end



% 1/26/23 load precision matrices for whole genome, normalize them so their
% inverse has unit diagonal, merge with baselineLD annotations, compute LD
% scores for selected annotations including null annotations.

clear;clc

whichChromosomes = 1:22;
population = 'EUR';

addpath(genpath('~/Dropbox/GitHub/ldgm/MATLAB/'));
addpath '~/Dropbox/GitHub/h2-ldgm/MATLAB'
addpath '~/Dropbox/GitHub/h2-ldgm/sparseinv'

% Expects subdirectories called 'ldgm' and
% '1000G_Phase3_baselineLD_v2.2_ldscores' containing precision matrices and
% .annot files, respectively
datapath = '/Volumes/T7/data/';

P = []; snplists = []; AF = []; whichIndicesAnnot = []; mergedAnnot = []; 
for chromosome = whichChromosomes
    ldgms_dir = [datapath, 'ldgm/1kg_chr',num2str(chromosome),'_'];
    annot_dir = [datapath, ...
        '1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.',...
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

% Select which annotations to retain
whichAnnot = [5:9, 14:16, 19:20, 40:43];
annotNames = mergedAnnot{1}.Properties.VariableNames(whichAnnot)';
disp(annotNames)

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

% LD scores, as a cell array of matrices
l2 = calculate_LDscores(P,annotMatrices,whichIndicesAnnot);




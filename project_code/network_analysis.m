%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Network structure tutorial
% IMPORTANT - THIS IS THE FULL VERSION - It has all the answers
% Please follow the intermediate or the advanced one during class!
% Authors - H. Benisty & MR
% June 1 2025, Neural Data Science 2025
addpath(genpath('..'))

close all; clear; clc; rng(123);
path2data = 'steinmetz_selected_data/';
%% pick a session
sesPath = 'Moniz_2017-05-16'; % session with both motor and sensory areas
%sesPath = 'Forssmann_2017-11-01'; % session with medial regions and HPC regions
%sesPath = 'Lederberg_2017-12-05'; % sessions with motor, sensory and caudate putamen
%% Read in spike data .. ~5 sec
% Note that regions are indexed 1 to regions.N but neurons are indexed Python-style from 0 to neurons.N-1
[S, regions, neurons, trials] = stOpenSession([path2data,sesPath]);  % load .npy files in which data stored
sessionTime = S.spikes.times(end); % total time, assuming start at 0

%% plot the average firing rates in LGd and VIsp
% Put all spikes into a cell array
clusters = unique(S.spikes.clusters);
spikeMatrix = cell(length(clusters), 1); 

for i = 1:length(clusters)
    cluster = clusters(i);
    idx = S.spikes.clusters == cluster;
    neuronSpikes = S.spikes.times(idx);
    spikeMatrix{i} = neuronSpikes; 
end

% need neurons x time_bins and the stack all the trials
binSize = 0.005;
timeWindow = [0 sessionTime];

%change window here
psthBins = timeWindow(1):binSize:timeWindow(2);
nBins = length(edges)-1;
psthCenters = psthBins(1:end-1) + diff(psthBins)/2;

nNeurons = size(spikeMatrix,1);

% concat PSTH
concatPSTH = zeros(nNeurons, length(psthBins)-1);

for i = 1:nNeurons
    % first need to bin the PSTH
    spikes = spikeMatrix(1);
    spikes = cell2mat(spikes);
    binnedCounts = histcounts(spikes, psthBins);
    concatPSTH(i,:) = binnedCounts;
end

% now concat PSTH is in neurons x time bins

% what is the form of the data needed for this?
% data is cells x time trials
% code labels is a cell array with the numbers 1-80
codeLabels = num2cell(1:nNeurons);
%% Step -1: fake data
% data is cells over time, compute correlations 
C = corr(concatPSTH');
% plot correlation matrix
figure;imagesc(C); setXLabels(codeLabels);
plotUndirectedCentrality(C, codeLabels);

%% Step 0: set path to data files
path2data = '../../data/Zebrafish/';
[dat, data] = zbLoadOneFish(path2data, 'subject_1');
dat = dat'; % dat is cells over time
%% Step 1: load meta data
trialLen = 30; % this is true for subject 1
% load code labels
[cellAnatCode, codeNums, codeLabels] = zbMakeAnatomicIndex(path2data, data);
%% Step 2: get the pca of all brain regions 
for brain_area_i = 1:length(codeLabels)
    disp(brain_area_i);
    x = double(dat(cellAnatCode == codeNums(brain_area_i), :));
    [~, x_pca{brain_area_i}, ev_x{brain_area_i}] = pca(x');
end
%% Step 3: undirected graphs
%% Step 3a: correlate 1PC across brain regions 
for brain_area_src = 1:length(codeLabels)
    src_pca = x_pca{brain_area_src};
    for brain_area_dst = 1:length(codeLabels)
         dst_pca = x_pca{brain_area_dst};
         C_corr_1pca(brain_area_src, brain_area_dst) = corr(src_pca(:, 1), dst_pca(:, 1));
    end
end
% plot centrality of brain regions using plotUndirectedCentrality
plotUndirectedCentrality(C_corr_1pca, codeLabels);
%% Step 3b: precision matrix
precision_matrix = pinv(C_corr_1pca);
% plot centrality 
plotUndirectedCentrality(precision_matrix, codeLabels);
%% Step 4: directed graphs
%% Step 4a: model 1st PC from one area using the first 1-3 of another
for brain_area_src = 1:length(codeLabels)
    eff_dim = 3;
    src_pca = x_pca{brain_area_src};
    for brain_area_dst = 1:length(codeLabels)
        dst_pca = x_pca{brain_area_dst};        
        lm = fitlm(src_pca(:, 1:eff_dim), dst_pca(:, 1));
        C_r_pca13(brain_area_src, brain_area_dst) = lm.Rsquared.Adjusted;
    end
end
plotDirectedCentrality(C_r_pca13, codeLabels);
%% Step 4b: model 1st PC from one area using 90% of another
for brain_area_src = 1:length(codeLabels)
    eff_dim = find(cumsum(ev_x{brain_area_src})/sum(ev_x{brain_area_src})>.9, 1);
    src_pca = x_pca{brain_area_src};
    for brain_area_dst = 1:length(codeLabels)
        dst_pca = x_pca{brain_area_dst};        
        lm = fitlm(src_pca(:, 1:eff_dim), dst_pca(:, 1));
        C_r_pca(brain_area_src, brain_area_dst) = lm.Rsquared.Adjusted;
    end
end
plotDirectedCentrality(C_r_pca, codeLabels);

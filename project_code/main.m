%% Explore various features (outcomes and behavior data) of a typical Steinmetz 2019 (2-probe) data set
% This script also does some useful book-keeping and summarizing that you
% may find useful in many later exercises
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
% save the binned tensor
save(['postprocessed_data/' sesPath '_S.mat'], 'S')
save(['postprocessed_data/' sesPath '_regions.mat'], 'regions')
save(['postprocessed_data/' sesPath '_neurons.mat'], 'neurons')
save(['postprocessed_data/' sesPath '_trials.mat'], 'trials')

% note which regions: for Moniz LS on probe 0; DG, SUB & CA3 on probe 1 ; ACA & MOs on probe 0; VISam on probe 1
regionTable = table( histcounts(neurons.region(neurons.probe==0),...
   .5:1:(regions.N-.5))', histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5))', ...
   'VariableNames',["Probe 0" "Probe 1"],'RowNames', regions.name(1:regions.N-1));
%We select LGd(3) and VISp(10)
regionSelected = [3,10];
colorSelected = ['r','b'];
%disp(regionTable) % print out region names
% fprintf('\nProbe 0 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==0),.5:1:(regions.N-.5)))
% fprintf('\nProbe 1 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5)))
% Initialize for data
sessionTime = S.spikes.times(end); % total time, assuming start at 0
stimTimes = trials.visStimTime;
respTimes = trials.responseTime;
goTimes = S.trials.goCue_times;
% construct logical variable for spike timestamps in trials
inTrial = false(size(S.spikes.times,1),1);
for kk = 1:trials.N
   inTrial( S.spikes.times > stimTimes(kk) & S.spikes.times < respTimes(kk) ) = true;
end

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

%% Plot average firing rate as histogram
figure()
for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = spikeMatrix(region_idx);
    % generate empty array
    firingRates = zeros(1,length(region_neurons)); 
    % find average firing rate of each neuron in this cell array
    for i = 1:length(region_neurons)
       spikes = region_neurons{i};
       numSpikes = length(spikes);
       firingRate = numSpikes/sessionTime;
       firingRates(i) = firingRate;
    end

    maxFR = max(firingRates);
    bins = 1:2:maxFR;
    histogram(firingRates, bins, 'EdgeColor', 'none', 'FaceColor',colorSelected(rr));
    hold on
    xlabel('Firing rates (spikes/second)')
    ylabel('Counts')
    title('Firing Rates in Selected Regions')
end
legend([regions.name(regionSelected)]);
saveas(gcf, ['figure/' sesPath '_firing rate.fig']);

%% Create PSTH and change into tensor format
% need neurons x time_bins x trials
binSize = 0.005;
timeWindow = [-0.5 2];
%change window here
edges = timeWindow(1):binSize:timeWindow(2);
nBins = length(edges)-1;
nTrials = length(stimTimes);
nNeurons = length(spikeMatrix);
% Preallocate output
binnedTensor = zeros(nNeurons, nBins, nTrials);
% Loop through neurons
for n = 1:nNeurons
   neuronSpikes = spikeMatrix{n};
   % Loop through trials
   for t = 1:nTrials       
       trialStart = stimTimes(t);
       trialSpikes= neuronSpikes(neuronSpikes >=trialStart + timeWindow(1) & ...
           neuronSpikes < trialStart + timeWindow(2));
       alignedSpikes = trialSpikes - trialStart;
       binnedCounts = histcounts(alignedSpikes, edges);
       binnedTensor(n,:,t) = binnedCounts;
   end
end
% Now have a neurons x PSTH x trials array

% save the binned tensor
save(['postprocessed_data/' sesPath '_binnedTensor.mat'], 'binnedTensor')

%% Plot tuning curves according to stimulus and firing rate
behavior = trials.contrast;%can change it
%behavior = trials.turn;
behavior_value = unique(behavior);
valueNum = length(behavior_value);

figure(1);
for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = binnedTensor(region_idx, :, :);

    summedTensor = sum(region_neurons,2);
    neuronNum = size(region_neurons,1);
    
    firing_Onbehavior = zeros(valueNum,neuronNum);
    time_Onbehavior = zeros(valueNum,neuronNum);

    for i = 1:neuronNum
        for j = 1:nTrials
            for v = 1:valueNum
                if behavior(j)==behavior_value(v)
                    firing_Onbehavior(v,i) = firing_Onbehavior(v,i)+summedTensor(i, 1, j);
                    time_Onbehavior(v,i) = time_Onbehavior(v,i)+1;
                end
            end
        end
    end

    firingRate_Onbehavior = firing_Onbehavior./time_Onbehavior;
    [peak_values, peak_stimuli] = max(firingRate_Onbehavior, [], 1);

    neuron_groups = cell(valueNum, 1);
    firingRate_maxnormalized = zeros(size(firingRate_Onbehavior));
    for i = 1:size(firingRate_Onbehavior,2)
        firingRate_maxnormalized(:,i) =  firingRate_Onbehavior(:,i)/max(firingRate_Onbehavior(:,i));
    end

    %sort the neuron by peak stimuli
    [sortedArray, idx] = sort(peak_stimuli);
    firingRate_maxnormalized_sorted = firingRate_maxnormalized(:,idx);
    newfigure  = figure;
    imagesc(firingRate_maxnormalized_sorted')
    %colorbar;  
    cb = colorbar;
    cb.Label.String = 'Normalized firing rate';  % 设置标签文本
    cb.Label.FontSize = 12;         % 调整字体大小[5,7](@ref)
    % 获取当前坐标轴
    ax = gca;
    
    set(ax, 'XTickLabel', compose('%.2f', behavior_value)); 
    xlabel('Visual contrast')
    ylabel('Sorted neurons')
    title(regions.name(region_code))
    
    saveas(newfigure, sprintf('figure/%s_tuning heatmap_%s.fig', sesPath, regions.name(region_code)));
    %saveas(newfigure,'test.fig');
    close(newfigure);

    %ylabel(regions.name(region_code))
    
    % Group neurons by their peak stimulus
    for stim = 1:valueNum
        %subplot(valueNum,2,rr+(stim-1)*2)
        subplot(2,valueNum,stim+(rr-1)*valueNum)
        neuron_groups{stim} = find(peak_stimuli == stim);
        p = 1;
        % check if the length is same
        if (~isempty(neuron_groups{stim}))
            %add a normalization by divide by their minimum
            firingRate_normalized = normalize(firingRate_Onbehavior(:,neuron_groups{stim}), 'range');
            %firingRate_normalized = firingRate_Onbehavior(:,neuron_groups{stim});
            
            plot(behavior_value,firingRate_normalized);
            %imagesc(firingRate_Onbehavior)
            hold on
            %add mean line
            meanRate = mean(firingRate_normalized,2); 
            plot(behavior_value,meanRate, 'k', 'LineWidth', 3);
            %Shapiro-Wilk Test 
            [h, p, W] = swtest(meanRate); 
            hold on
            xticks(behavior_value);
        end
        if p<0.05
            title(['neuronum = ',num2str(size(firingRate_Onbehavior(:,neuron_groups{stim}),2)),' *'])
        else
            title(['neuronum = ',num2str(size(firingRate_Onbehavior(:,neuron_groups{stim}),2))])
        end
        ylabel('Normalized firing rate')
        xlabel('Visual contrast')
    
    end    
    %xlabel(regions.name(region_code))
    annotation('textbox', [0.4, 0.95 - (rr-1)*0.5, 0.2, 0.05], ...
              'String', regions.name(region_code), 'EdgeColor', 'none', ...
              'HorizontalAlignment', 'center', 'FontSize', 12);

end
% normalize the tuning curve
%saveas(gcf, ['figure/' sesPath '_tuning curve_normalized.fig']);
saveas(gcf, ['figure/' sesPath '_tuning curve_original.fig']);






%% Plot traces relative to stim, response, and go
% We already have two regions and we need to plot them together. Make it
% parameters.
% vars are stimTimes, respTimes, goTimes

% Smooth across time
smoothedTensor = movmean(binnedTensor, [5 5], 2);
means = [];

figure;
t = tiledlayout(3, 2);  % 3行2列网格

for rr = 1:2
    nexttile(rr)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = smoothedTensor(region_idx, :, :);

    % Plot all neurons, mean of all trials
    for i = 1:sum(region_idx)
       % average trial response
       avg_resp = mean(region_neurons(i,:,:), 3);
       plot(avg_resp)
       means(i,:) = avg_resp;
       hold on
    end
    mean_resp = mean(means);
    xlabel('Time bins')
    ylabel('Response amplitude')
    title(regions.name(regionSelected(rr)))
    plot(mean_resp, 'k', 'LineWidth', 3);
    hold off
end

%% Reduce the values of trial for PCA
stepSize = 10;
idx = 1:stepSize:size(smoothedTensor,2);

tensorPCA = smoothedTensor(:,idx,:);
%allScores = [];
for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);

    % Generate PCA
    nexttile(rr+2)

    % find
    averageTrials = mean(region_neurons,3);

    % Run the PCA
    [coefs, scores, ~, ~, explained ] = pca(averageTrials');
    cumulative_variance = cumsum(explained)/100;
    K = find(cumulative_variance >= 0.80, 1); % choose K

    %plot Scree-plot to decide K
    plot(1:length(explained), explained, 'bo-', 'LineWidth', 2);
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    title(['Scree Plot for PCA on',regions.name(regionSelected(rr))],[num2str(K),' component count for 80% variance']);
    xlim([0 10])
    grid on;
    %allScores(:,:,rr) = scores;
    % separate pca; can't draw together
    nexttile(rr+4)
    plot3(scores(:,1), scores(:,2), scores(:,3),colorSelected(rr));
    xlabel('PCA 1')
    ylabel('PCA 2')
    zlabel('PCA 3')
    title('PCA on',regions.name(region_code))
 
end
% nexttile([1, 2]);
% for rr = 1:2
%     % plot coef by signals
%     plot3(allScores(:,1,rr), allScores(:,2,rr), allScores(:,3,rr),colorSelected(rr));
%     hold on
% end
% legend([regions.name(regionSelected)]);
saveas(gcf, ['figure/' sesPath '_pca.fig']);
%% NMF
nFactors = 5;
figure;
t = tiledlayout(1, 2); 

for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);
     % Generate NMF
    nexttile(rr)

    % find
    averageTrials = mean(region_neurons,3);
   
    % this is the nmf
    [W,H, disc] = nnmf(averageTrials, nFactors); % specify six factors
    myNMFLoads = H';
    % plot coef by signals
    plot3(myNMFLoads(:,1), myNMFLoads(:,2), myNMFLoads(:,3),colorSelected(rr));
    xlabel('C 1'); ylabel('C 2'); zlabel('C 3');
    title('NMF on',regions.name(region_code))
    hold on

end

saveas(gcf, ['figure/' sesPath '_nmf.fig']);




%% Umap
% Use 'run_umap' to reduce the dim to 3
% try different values for n_neighbors ranging from 5 to 199
n_components = 3;
n_neighbors = 10;
figure;
t = tiledlayout(1, 2); 

for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);
     % Generate UMAP
    nexttile(rr)

    % find
    averageTrials = mean(region_neurons,3);
    [rep_UMAP, umap, clusterIdentifiers, extras]=run_umap(double(averageTrials'), ...
    'n_components', n_components, 'n_neighbors', n_neighbors, 'verbose', 'none');
    % plot coef by signals
    plot3(rep_UMAP(:,1), rep_UMAP(:,2), rep_UMAP(:,3),colorSelected(rr));
    xlabel('UMAP 1')
    ylabel('UMAP 2')
    zlabel('UMAP 3')
    title('UMAP on',regions.name(region_code))
    hold on

end

saveas(gcf, ['figure/' sesPath '_umap.fig']);


%% Find trials where each stimulus occurs
leftStim = S.trials.visualStim_contrastLeft;
rightStim = S.trials.visualStim_contrastRight;

% create stimtype var
stimType = zeros(1,length(leftStim));

% create vector that has left as 1 and 
% right as 2
% both as 3
% none as 0
for i = 1:length(leftStim)
    if leftStim(i) > 0 && rightStim(i) > 0
        stimType(i) = 3;
    elseif leftStim(i) > 0 && rightStim(i) == 0
        stimType(i) = 1;
    elseif leftStim(i) == 0 && rightStim(i) > 0
        stimType(i) = 2;
    else stimType(i) = 0;
    end
end


%% Plot PCA for each trial type

%% Explore various features (outcomes and behavior data) of a typical Steinmetz 2019 (2-probe) data set
% This script also does some useful book-keeping and summarizing that you
% may find useful in many later exercises
path2data = 'C:\cshl-neudata-2025\Steinmetz_raw\steinmetz_project\';
%% pick a session
sesPath = 'Moniz_2017-05-16'; % session with both motor and sensory areas
%sesPath = 'Forssmann_2017-11-01'; % session with medial regions and HPC regions
%sesPath = 'Lederberg_2017-12-05'; % sessions with motor, sensory and caudate putamen
%% Read in spike data .. ~5 sec
% Note that regions are indexed 1 to regions.N but neurons are indexed Python-style from 0 to neurons.N-1
[S, regions, neurons, trials] = stOpenSession([path2data,sesPath]);  % load .npy files in which data stored
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
    

    % subplot(2,2,rr+2)
    % % preallocate mean array
    % means = zeros(length(region_neurons));
    % % Plot all neurons, mean of all trials
    % for i = 1:sum(region_idx)
    %    % average trial response
    %    avg_resp = mean(region_neurons(i,:), 2);
    %    plot(avg_resp)
    %    hold on
    % end
end
legend([regions.name(regionSelected)]);

%% Create PSTH and change into tensor format
% need neurons x time_bins x trials
binSize = 0.005;
timeWindow = [-0.5 2];
%change window here
edges1 = timeWindow(1):binSize:timeWindow(2);
% edges2 = binSize/2:binSize:timeWindow(2);

% nBins = length(edges1)+length(edges2)-2;
nBins = length(edges1)-1;
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
       binnedCounts1 = histcounts(alignedSpikes, edges1);
       % binnedCounts2 = histcounts(alignedSpikes, edges2);
  
       % total_counts = zeros(1,size(binnedCounts1,2)+size(binnedCounts2,2));
       % interleaved counts
       % total_counts(1:2:end) = binnedCounts1;
       % total_counts(2:2:end) = binnedCounts2;

       % binnedTensor(n,:,t) = total_counts;
       binnedTensor(n,:,t) = binnedCounts1;
   end
end
% Now have a neurons x PSTH x trials array
%% Plot traces relative to stim, response, and go
% We already have two regions and we need to plot them together. Make it
% parameters.
% vars are stimTimes, respTimes, goTimes

% Smooth across time
smoothedTensor = movmean(binnedTensor, [5 5], 2);
means = [];

figure()
for rr = 1:2
    subplot(3,2,rr)
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
    plot(mean_resp, 'k', 'LineWidth', 3);
    hold off
end

%% Reduce the values of trial for PCA
stepSize = 10;
idx = 1:stepSize:size(smoothedTensor,2);

tensorPCA = smoothedTensor(:,idx,:);

for rr = 1:2
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);

    % Generate PCA
    subplot(3,2,rr+2)

    % find
    averageTrials = mean(region_neurons,3);

    % Run the PCA
    [coefs, scores, ~, ~, explained ] = pca(averageTrials',"NumComponents",3);
    cumulative_variance = cumsum(explained)/100;
    K = find(cumulative_variance >= 0.80, 1); % 自动选择K

    %plot Scree-plot to decide K
    plot(1:length(explained), explained, 'bo-', 'LineWidth', 2);
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    title(['Scree Plot for PCA on',regions.name(regionSelected(rr))],[num2str(K),' component count for 80% variance']);
    xlim([0 10])
    grid on;

    % plot coef by signals
    subplot(3,2,rr+4)
    plot3(scores(:,1), scores(:,2), scores(:,3));
end


%% Find trials where left stimulus only
S.trials.VisualStim_contrastLeft




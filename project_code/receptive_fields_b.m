%% Explore various features (outcomes and behavior data) of a typical Steinmetz 2019 (2-probe) data set
% This script also does some useful book-keeping and summarizing that you
% may find useful in many later exercises

%% pick a session 
path2data = 'C:\cshl-neudata-2025\Steinmetz_raw\steinmetz_project\';
sesPath = 'Moniz_2017-05-16'; % session with both motor and sensory areas
%sesPath = 'Forssmann_2017-11-01'; % session with medial regions and HPC regions
%sesPath = 'Lederberg_2017-12-05'; % sessions with motor, sensory and caudate putamen

%% Read in spike data .. ~5 sec
% Note that regions are indexed 1 to regions.N but neurons are indexed Python-style from 0 to neurons.N-1
[S, regions, neurons, trials] = stOpenSession([path2data sesPath]);  % load .npy files in which data stored
% note which regions: for Moniz LS on probe 0; DG, SUB & CA3 on probe 1 ; ACA & MOs on probe 0; VISam on probe 1
regionTable = table( histcounts(neurons.region(neurons.probe==0),...
    .5:1:(regions.N-.5))', histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5))', ...
    'VariableNames',["Probe 0" "Probe 1"],'RowNames', regions.name(1:regions.N-1));
disp(regionTable) % print out region names
% fprintf('\nProbe 0 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==0),.5:1:(regions.N-.5)))
% fprintf('\nProbe 1 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5)))
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
spikeMatrix = {};
z = 1;

for i = 1:length(clusters)
    cluster = clusters(i);
    
    % create mask;
    idx = S.spikes.clusters == cluster;

    % apply mask
    neuronSpikes = S.spikes.times(idx);

    % add to array
    spikeMatrix{z} = neuronSpikes;

    % update counter
    z = z+1;
end


%% Create PSTH and change into tensor format
% need neurons x time_bins x trials
binSize = 0.010;
timeWindow = [-0.5 2];
edges = timeWindow(1):binSize:timeWindow(2);
nBins = length(edges) - 1;

nTrials = length(stimTimes);
nNeurons = length(spikeMatrix);

% Preallocate output
binnedTensor = zeros(nNeurons, nBins, nTrials);

% Loop through neurons
for n = 1:nNeurons
    neuronSpikes = spikeMatrix{n};

    % Loop through trials
    for t = 1:nTrials
        total_counts = [];
        trialStart = stimTimes(t);
        trialSpikes= neuronSpikes(neuronSpikes >=trialStart + timeWindow(1) & ...
            neuronSpikes < trialStart + timeWindow(2));
        alignedSpikes = trialSpikes - trialStart;
        binnedCounts = histcounts(alignedSpikes, edges);
        binnedTensor(n,:,t) = binnedCounts;

    end
end

% Now have a neurons x PSTH x trials array
smoothedTensor = movmean(binnedTensor, 10, 3);

%% Plot traces relative to stim, response, and go
% vars are stimTimes, respTimes, goTimes
region_code = 3; % LGd
LGD_idx = neurons.region == region_code;
LGD_neurons = smoothedTensor(LGD_idx, :, :);

% preallocate mean array
means = [];

% Plot all neurons, mean of all trials
for i = 1:sum(LGD_idx)
    % average trial response
    avg_resp = mean(LGD_neurons(i,:,:), 3);
    plot(avg_resp)
    means(i,:) = avg_resp;
    hold on
end
plot(means, 'k')

%% Generate PCA
% find 
averageTrials = mean(LGD_neurons,3);

% Run the PCA
[coefs, scores, ~, ~, explained ] = pca(averageTrials',"NumComponents",3);

cumulative_variance = cumsum(explained)/100;
K = find(cumulative_variance >= 0.80, 1); % 自动选择K
%plot Scree-plot to decide K

plot(1:length(explained), explained, 'bo-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
grid on;

% plot coef by signals
plot3(scores(:,1), scores(:,2), scores(:,3));
%% Plot average firing rate for VIsp as histogram
figure()

% Find cells that belong to VIsp
region_code = 10; % VIsp
% region_code = 3 % LGd
VIsp_idx = neurons.region == region_code;
VIsp_neurons = spikeMatrix(VIsp_idx);

% generate empty array
firingRates = [];

% find average firing rate of each neuron in this cell array
for i = 1:length(VIsp_neurons)
    spikes = VIsp_neurons{i};
    numSpikes = length(spikes);
    firingRate = numSpikes/sessionTime;
    VIspfiringRates(i) = firingRate;
end

%% Plot average firing rate for LGd as histogram
% Find cells that belong to LGD
region_code = 3; % LGd
LGD_idx = neurons.region == region_code;
LGD_neurons = spikeMatrix(LGD_idx);

% generate empty array
LGdfiringRates = [];

% find average firing rate of each neuron in this cell array
for i = 1:length(LGD_neurons)
    spikes = LGD_neurons{i};
    numSpikes = length(spikes);
    firingRate = numSpikes/sessionTime;
    LGdfiringRates(i) = firingRate;
end

maxFR = max(LGdfiringRates);
bins = 1:2:maxFR;

% Plot histogram
figure()
subplot(1, 2, 1)
histogram(VIspfiringRates, bins, 'EdgeColor', 'none', 'FaceColor', 'm');
hold on
xlabel('Firing rates (spikes/second)')
ylabel('Counts')
title('Firing Rates in VIsp')
hold off
subplot(1, 2, 2)
histogram(LGdfiringRates, bins, 'EdgeColor', 'none', 'FaceColor', 'b');
hold on
title('Firing Rates in LGd')
xlabel('Firing rates (spikes/second)')
ylabel('Counts')
hold off

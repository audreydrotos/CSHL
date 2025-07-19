%% build a model that predicts neural firing rates in your target 
% region as well as possible (maximize R^2) without overfitting.
% There are all kinds of predictors available for this purpose 
% (visual inputs, motor responses (eyes, face), and a plethora of 
% other covariates (e.g. licks).

%% From other analysis
% This script also does some useful book-keeping and summarizing that you
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
% Display table showing regions in this data set
regionTable = table( histcounts(neurons.region(neurons.probe==0),...
    .5:1:(regions.N-.5))', histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5))', ...
    'VariableNames',["Probe 0" "Probe 1"],'RowNames', regions.name(1:regions.N-1));
disp(regionTable) % print out region names
% fprintf('\nProbe 0 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==0),.5:1:(regions.N-.5)))
% fprintf('\nProbe 1 region counts: %d %d %d %d %d %d %d %d %d',histcounts(neurons.region(neurons.probe==1),.5:1:(regions.N-.5)))
%% Book-keeping
sessionTime = S.spikes.times(end); % total time, assuming start at 0
stimTimes = trials.visStimTime; 
respTimes = trials.responseTime;
goTimes = S.trials.goCue_times;
% construct logical variable for spike timestamps in trials
inTrial = false(size(S.spikes.times,1),1);
for kk = 1:trials.N
    inTrial( S.spikes.times > stimTimes(kk) & S.spikes.times < respTimes(kk) ) = true;
end

%% Book-keeping
neuronNumEdges = (0:neurons.N) - 0.5; % edges for using histcounts to count neuron cluster IDs in spikes

%% Parameters to convert behavior and DLC motion-capture frames to time in sec 
faceframe2timeInt = S.face.timestamps(1,2); % usually 10 - 20 sec after recording start
faceframe2timeSlope = (S.face.timestamps(2,2)-S.face.timestamps(1,2))/(S.face.timestamps(2,1)-S.face.timestamps(1,1)); % usually ~ 1/40Hz
% Note: DeepLabCut variables are on same time as face camera 
eyeframe2timeInt = S.eye.timestamps(1,2); % usually 10 - 20 sec after recording start
eyeframe2timeSlope = (S.eye.timestamps(2,2)-S.eye.timestamps(1,2))/(S.eye.timestamps(2,1)-S.eye.timestamps(1,1)); % usually ~ 1/100Hz
S.eye.area( S.eye.area < 0) = 0; % there are some glitches giving negative values; 
% NB you'll need to filter the eye data
wheelframe2timeInt = S.wheel.timestamps(1,2); % usually 10 - 20 sec after recording start
wheelframe2timeSlope = (S.wheel.timestamps(2,2)-S.wheel.timestamps(1,2))/(S.wheel.timestamps(2,1)-S.wheel.timestamps(1,1)); % ~ 1/2500Hz

%% Explore trial response statistics
histogram( respTimes-stimTimes,15)
% after 1.5 sec have elapsed with no turn the response is recorded as 0
nn=find(respTimes - S.trials.goCue_times > 1.5); % typical 'cut-off' for trials is around 1.5 sec - a bit of jitter
histcounts( S.trials.response_choice(nn)) % all 'no choice' - timed out trials 
nn1=find( S.trials.response_choice == 0 & S.trials.visualStim_contrastLeft ~= S.trials.visualStim_contrastRight);
% some between trials 95 and 120 but most near the end 160-215; misses more R high contrast
nn2=find( S.trials.response_choice == 1 & S.trials.visualStim_contrastLeft < S.trials.visualStim_contrastRight); % wrong turns
% also some between trials 95 and 120 but most near the end 160-215
nn3=find( S.trials.response_choice == -1 & S.trials.visualStim_contrastLeft > S.trials.visualStim_contrastRight); % wrong turns
isCorrectTrial = trials.Correct;
% 
figure
plot(stimTimes, respTimes - S.trials.goCue_times,'o'); xlabel('Stim times (sec)'); ylabel('Response - Go interval')
hold on; grid
plot(stimTimes(~isCorrectTrial), respTimes(~isCorrectTrial) - S.trials.goCue_times(~isCorrectTrial),'.r','MarkerSize',7); 
hold off
title('Trial times and durations: incorrect responses with red dot')
% eye motions
% plot x,y positions with diameter indicated & direction after onset of stimulus

%% Extract and explore behavior correlates provided by Nick Steinmetz
% Note: may get error if eye missing from data store
T1 = stimTimes(end)-100; T2 = stimTimes(end)+100; % period bracketing end of trials
% figure; 
% scatter(S.eye.xyPos(:,1),S.eye.xyPos(:,2),'.'); title('Eye Pos & Diameter'); grid % centered at 0,0
figure 
plot(S.eye.timestamps(:,2), smooth(S.eye.xyPos(:,1),9)); ylim([-3 7]); grid; hold
plot(S.eye.timestamps(:,2), smooth(S.eye.xyPos(:,2),9), 'color',[.1 .8 .2]) 
plot(S.eye.timestamps(:,2), sqrt(S.eye.area/pi)); ylim([0 1.5]);
line([ stimTimes'; stimTimes'], [-2*ones(1,length(stimTimes)) ;2*ones(1,length(stimTimes))],'color',[.5 .5 .5],'linestyle',':','linewidth',2)
title('Gaze direction & pupil size'); 
legend({'X','Y','D','Stim'})
xlim([ T1 T2]); xlabel('Time (sec)'); % can expand 
hold off % often some glitches (abrupt displacement and return in a few frames)

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

% Spike matrix now has all of the cells, need to subset by region
region_code = 3; % for LGD or 10 for VIsp

region_idx = neurons.region == region_code;
region_neurons = spikeMatrix(region_idx);

%% What behavioral correlates do we want to use for the GLM?
% First we need to put everything on the same clock, use the lowest HZ
% sampling rate?

% need neurons x time_bins x trials
binSize = 0.010;
timeWindow = [0 sessionTime];

%change window here
psthBins = timeWindow(1):binSize:timeWindow(2);
nBins = length(edges)-1;
psthCenters = psthBins(1:end-1) + diff(psthBins)/2;

% these are our X values
% face energy
nSamples = length(S.face.motionEnergy);
energyTimes = linspace(S.face.timestamps(1,2), S.face.timestamps(2,2), nSamples);
binnedFaceEnergy = interp1(energyTimes, S.face.motionEnergy, psthCenters, 'linear', 'extrap'); % interpolate

% eye energy
nSamples = length(S.eye.area);
eyeTimes = linspace(S.eye.timestamps(1,2), S.face.timestamps(2,2), nSamples);
binnedEyeEnergy = interp1(eyeTimes, S.eye.area, psthCenters, 'linear', 'extrap');

% visual inputs
S.trials.contrast;
stimTimes;

% licks
S.licks.times;

% this is our Y value--spike times, and then we will predict for every
% neuron separately. right now this is in region_neurons. will need to
% generate a PSTH live.
nNeurons = length(region_neurons);

%% for the PSTH
% create empty array to put together all the variables
B_array = zeros(1, nNeurons);
dev_array = zeros(1, nNeurons);
stats_array = [];

for i = 1:nNeurons
    % first need to bin the PSTH
    spikes = region_neurons(1);
    spikes = cell2mat(spikes);
    binnedCounts = histcounts(spikes, edges);
    Y = binnedCounts;

    % concat all predictors
    X = [binnedEyeEnergy', binnedFaceEnergy'];

    % fit the GLM
    mdl = fitglm(X, Y);

end



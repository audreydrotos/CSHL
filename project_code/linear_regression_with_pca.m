%% Linear Regression with PCA Analysis - Project Part 3 (CORRECTED VERSION)
% Build linear regression model to predict neural firing rates from behavioral data
% This version fixes data alignment and feature extraction issues

addpath(genpath('..'))
close all; clear; clc; 
rng(123); % Set random seed for reproducible results

% Disable GLM-related warnings globally
warning('off', 'stats:glmfit:BadScaling');
warning('off', 'stats:LinearModel:RankDefDesignMat');
warning('off', 'stats:glmfit:IterationLimit');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:rankDeficientMatrix');

%% ======================== 1. Data Loading and Basic Setup ========================
fprintf('Loading data...\n');

% Select session
sesPath = 'Moniz_2017-05-16'; % Session with motor and sensory areas
path2data = 'steinmetz_selected_data/';

% Load preprocessed data
if exist(['postprocessed_data/' sesPath '_binnedTensor.mat'], 'file')
    load(['postprocessed_data/' sesPath '_binnedTensor.mat'], 'binnedTensor');
    load(['postprocessed_data/' sesPath '_S.mat'], 'S');
    load(['postprocessed_data/' sesPath '_regions.mat'], 'regions');
    load(['postprocessed_data/' sesPath '_neurons.mat'], 'neurons');
    load(['postprocessed_data/' sesPath '_trials.mat'], 'trials');
    fprintf('Loaded preprocessed data.\n');
else
    error('Preprocessed data not found. Please run main.m first to generate the data.');
end

% Select early and late processing regions
regionSelected = [3, 10]; % LGd (early) and VISp (late)
regionNames = {'LGd', 'VISp'};
fprintf('Selected brain regions: %s (early) and %s (late)\n', regionNames{1}, regionNames{2});

%% ======================== 2. Trial-Based Feature Extraction ========================
fprintf('Preparing trial-based feature matrix...\n');

% Use the existing binnedTensor which is already aligned to trials
% binnedTensor: [nNeurons x nTimeBins x nTrials]
[nNeurons, nTimeBins, nTrials] = size(binnedTensor);
fprintf('Data dimensions: %d neurons, %d time bins, %d trials\n', nNeurons, nTimeBins, nTrials);

% Smooth across time
smoothedTensor = movmean(binnedTensor, [5 5], 2);

% Calculate average responses across trials for each neuron
neuralAvgResponses = struct();
means = [];

figure
for rr = 1:length(regionSelected)
    nexttile(rr)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = smoothedTensor(region_idx, :, :);
    
    % Store average responses for this region
    avg_responses_region = [];
    
    % Plot all neurons, mean of all trials
    means = [];
    for i = 1:sum(region_idx)
       % average trial response
       avg_resp = mean(region_neurons(i,:,:), 3);
       avg_responses_region(i, :) = avg_resp;
       plot(avg_resp)
       means(i,:) = avg_resp;
       hold on
    end
    
    % Store the average responses for this region
    neuralAvgResponses.(regionNames{rr}) = avg_responses_region;
    
    mean_resp = mean(means);
    xlabel('Time bins')
    ylabel('Response amplitude')
    title(regions.name(regionSelected(rr)))
    plot(mean_resp, 'k', 'LineWidth', 3);
    hold off
end

% Trial-based behavioral features (will be used for all time bins)
trial_features = [];
feature_names = {};

% 1. Visual stimulus features (trial-level)
fprintf('  - Extracting trial-level visual features\n');
leftContrast = S.trials.visualStim_contrastLeft;
rightContrast = S.trials.visualStim_contrastRight;
totalContrast = leftContrast + rightContrast;
contrastDiff = abs(leftContrast - rightContrast);
maxContrast = max(leftContrast, rightContrast);

% Ensure all are column vectors
leftContrast = leftContrast(:);
rightContrast = rightContrast(:);
totalContrast = totalContrast(:);
contrastDiff = contrastDiff(:);
maxContrast = maxContrast(:);

trial_features = [trial_features, leftContrast, rightContrast, totalContrast, contrastDiff, maxContrast];
feature_names = [feature_names, {'leftContrast', 'rightContrast', 'totalContrast', 'contrastDiff', 'maxContrast'}];

% 2. Behavioral choice features
fprintf('  - Extracting behavioral choice features\n');
choices = S.trials.response_choice; % -1, 0, 1 for left, no-go, right
choices(isnan(choices)) = 0; % Replace NaN with no-go
responseTime = S.trials.response_times - S.trials.goCue_times; % Reaction time
responseTime(isnan(responseTime) | responseTime < 0) = median(responseTime, 'omitnan'); % Fix invalid times
feedbackType = S.trials.feedbackType; % 1 for reward, -1 for no reward

% Ensure all are column vectors
choices = choices(:);
responseTime = responseTime(:);
feedbackType = feedbackType(:);

trial_features = [trial_features, choices, responseTime, feedbackType];
feature_names = [feature_names, {'choice', 'reactionTime', 'feedback'}];

% 3. Context features
fprintf('  - Extracting contextual features\n');
trialNumber = (1:nTrials)' / nTrials; % Normalized trial number (column vector)
blockType = mod(1:nTrials, 50)' / 50; % Block structure (column vector)

trial_features = [trial_features, trialNumber, blockType];
feature_names = [feature_names, {'trialNumber', 'blockType'}];

% 4. Previous trial features (history effects)
fprintf('  - Extracting trial history features\n');
prevChoice = [0; choices(1:end-1)];
prevFeedback = [0; feedbackType(1:end-1)];
prevContrast = [0; totalContrast(1:end-1)];

trial_features = [trial_features, prevChoice, prevFeedback, prevContrast];
feature_names = [feature_names, {'prevChoice', 'prevFeedback', 'prevContrast'}];

fprintf('Extracted %d trial-level features\n', size(trial_features, 2));

%% ======================== 3. Feature Matrix for Time-Point-wise Prediction ========================
fprintf('Creating feature matrix for time-point-wise prediction...\n');

% For time-point-wise prediction, we use trial-level behavioral features
% to predict neural activity at each specific time point
% Train/test split is across trials, not time

% Use the original trial-level behavioral features (no time expansion needed)
X_behavioral = trial_features; % [nTrials x nFeatures]
fprintf('Behavioral feature matrix: %d trials x %d features\n', size(X_behavioral));

% Standardize behavioral features
X_standardized = zscore(X_behavioral);
X_standardized(isnan(X_standardized)) = 0; % Handle NaN values

fprintf('Behavioral features used for prediction:\n');
for i = 1:length(feature_names)
    fprintf('  - %s\n', feature_names{i});
end

%% ======================== 4. PCA Feature Extraction ========================
fprintf('Performing PCA feature extraction on behavioral data...\n');

% Perform PCA on standardized behavioral features
[coeff, score, latent, ~, explained] = pca(X_standardized);

% Select components explaining 85% of variance
cumVar = cumsum(explained);
nComponents = find(cumVar >= 85, 1);
if isempty(nComponents)
    nComponents = min(10, size(X_standardized, 2)); % Fallback
end
fprintf('Selected first %d components (explaining %.1f%% variance)\n', nComponents, cumVar(nComponents));

% Create feature sets for behavioral data
featureSets = struct();
featureSets.raw = X_standardized;
featureSets.pca = score(:, 1:nComponents);
featureSets.combined = [X_standardized, score(:, 1:nComponents)];

%% ======================== 5. Neural Data Organization by Time Points ========================
fprintf('Organizing neural data by time points...\n');

% Neural responses organized by time points: binnedTensor is [nNeurons x nTimeBins x nTrials]
% We need to predict neural activity at each time point using behavioral features

% Filter neurons with very low activity for each region
fprintf('Filtering low-activity neurons...\n');
minFiringRate = 0.1; % Minimum firing rate threshold (Hz)
maxZeroFraction = 0.95; % Maximum fraction of zero trials allowed per neuron

neuralDataByRegion = struct();

for rr = 1:length(regionSelected)
    regionCode = regionSelected(rr);
    regionName = regionNames{rr};
    
    % Get neurons in this region
    regionIdx = neurons.region == regionCode;
    if sum(regionIdx) == 0
        warning('No neurons found in region %s', regionName);
        continue;
    end
    
    Y_region_raw = binnedTensor(regionIdx, :, :); % [nRegionNeurons x nTimeBins x nTrials]
    nRegionNeurons = size(Y_region_raw, 1);
    
    fprintf('\nRegion %s: %d neurons before filtering\n', regionName, nRegionNeurons);
    
    % Calculate statistics across trials for filtering
    Y_region_2d = reshape(Y_region_raw, nRegionNeurons, []); % [nNeurons x (nTimeBins*nTrials)]
    firingRates = mean(Y_region_2d, 2); % Mean across all time bins and trials
    zeroFractions = mean(Y_region_2d == 0, 2); % Fraction of zero samples
    
    fprintf('  Data distribution analysis:\n');
    fprintf('  - Mean firing rate: %.3f ± %.3f Hz\n', mean(firingRates), std(firingRates));
    fprintf('  - Firing rate range: %.3f - %.3f Hz\n', min(firingRates), max(firingRates));
    fprintf('  - Mean zero fraction: %.3f ± %.3f\n', mean(zeroFractions), std(zeroFractions));
    
    % Filter neurons
    validNeurons = find(zeroFractions < maxZeroFraction & firingRates >= minFiringRate);
    
    % If too few neurons pass, relax criteria
    if length(validNeurons) < 5
        fprintf('  Too few neurons passed initial criteria, relaxing...\n');
        validNeurons = find(firingRates > 0.01 & zeroFractions < 0.99);
    end
    
    if isempty(validNeurons)
        warning('No valid neurons found in region %s after filtering', regionName);
        continue;
    end
    
    Y_region_filtered = Y_region_raw(validNeurons, :, :); % [nValidNeurons x nTimeBins x nTrials]
    fprintf('Region %s: %d neurons after filtering\n', regionName, length(validNeurons));
    
    neuralDataByRegion.(regionName) = Y_region_filtered;
    fprintf('  Final data shape: %d neurons x %d time bins x %d trials\n', size(Y_region_filtered));
end

%% ======================== 6. Cross-Validation Setup ========================
fprintf('Setting up cross-validation for time-point-wise prediction...\n');

% For time-point-wise prediction, we split trials into train/test sets
% Each model is trained on behavioral data to predict neural activity at one specific time point
nFolds = 5;
nTrialsTotal = size(X_standardized, 1);

% Create trial-based cross-validation indices manually
rng(42); % Set random seed for reproducible results
randTrials = randperm(nTrialsTotal); % Random permutation of trial indices
foldSize = floor(nTrialsTotal / nFolds);

cv_indices = zeros(nTrialsTotal, 1);
for k = 1:nFolds
    if k < nFolds
        startIdx = (k-1) * foldSize + 1;
        endIdx = k * foldSize;
    else
        % Last fold gets remaining trials
        startIdx = (k-1) * foldSize + 1;
        endIdx = nTrialsTotal;
    end
    cv_indices(randTrials(startIdx:endIdx)) = k;
end

fprintf('Using trial-based CV: %d folds, %d trials total\n', nFolds, nTrialsTotal);
fprintf('Fold sizes: %s\n', mat2str(histcounts(cv_indices, 1:nFolds+1)));

%% ======================== 7. Time-Point-wise Model Training ========================
fprintf('Starting time-point-wise model training and evaluation...\n');

% Model types (removed GLM models)
modelTypes = {'ridge', 'lasso'};
featureTypes = fieldnames(featureSets);

% Results storage
results = struct();
predictionResults = struct(); % Store predictions for visualization

% Create time axis for the analysis
timeStart = -0.5;
timeEnd = 2.5;
time_axis = linspace(timeStart, timeEnd, nTimeBins);

% Train models for each brain region
for rr = 1:length(regionNames)
    regionName = regionNames{rr};
    
    if ~isfield(neuralDataByRegion, regionName)
        continue;
    end
    
    fprintf('\n=== Analyzing region: %s ===\n', regionName);
    Y_region = neuralDataByRegion.(regionName); % [nNeurons x nTimeBins x nTrials]
    [nRegionNeurons, nTimeBins, nTrials] = size(Y_region);
    
    results.(regionName) = struct();
    predictionResults.(regionName) = struct();
    
    % Train models for each feature set and model type
    for ft = 1:length(featureTypes)
        featureType = featureTypes{ft};
        X = featureSets.(featureType); % [nTrials x nFeatures]
        
        fprintf('Feature type: %s (%d trials x %d features)\n', featureType, size(X, 1), size(X, 2));
        
        results.(regionName).(featureType) = struct();
        predictionResults.(regionName).(featureType) = struct();
        
        for mt = 1:length(modelTypes)
            modelType = modelTypes{mt};
            fprintf('  Model type: %s\n', modelType);
            
            % Initialize storage for time-point predictions
            timePointR2 = zeros(nTimeBins, nFolds);
            timePointMSE = zeros(nTimeBins, nFolds);
            
            % Store predictions for visualization (using first fold)
            actualData = [];
            predictedData = [];
            
            % Cross-validation across trials
            for fold = 1:nFolds
                testTrials = find(cv_indices == fold);
                trainTrials = find(cv_indices ~= fold);
                
                X_train = X(trainTrials, :);
                X_test = X(testTrials, :);
                
                % Train and test model at each time point
                for t = 1:nTimeBins
                    % Get neural data for this time point across neurons and trials
                    % Average across neurons for simplicity (or could do for each neuron)
                    
                    % Extract neural data safely with proper dimension handling
                    if nRegionNeurons == 1
                        % Special case: only one neuron
                        Y_t_train_raw = squeeze(Y_region(1, t, trainTrials)); % [nTrainTrials]
                        Y_t_test_raw = squeeze(Y_region(1, t, testTrials));   % [nTestTrials]
                    else
                        % Multiple neurons: average across them
                        Y_t_train_raw = squeeze(mean(Y_region(:, t, trainTrials), 1)); % [nTrainTrials]
                        Y_t_test_raw = squeeze(mean(Y_region(:, t, testTrials), 1));   % [nTestTrials]
                    end
                    
                    % Ensure column vectors regardless of input dimensions
                    Y_t_train = Y_t_train_raw(:); % Force column vector [nTrainTrials x 1]
                    Y_t_test = Y_t_test_raw(:);   % Force column vector [nTestTrials x 1]
                    
                    % Verify dimensions
                    if size(Y_t_train, 1) ~= size(X_train, 1)
                        fprintf('    Dimension mismatch at time point %d: Y_train %s vs X_train %s\n', ...
                            t, mat2str(size(Y_t_train)), mat2str(size(X_train)));
                        continue;
                    end
                    
                    if size(Y_t_test, 1) ~= size(X_test, 1)
                        fprintf('    Dimension mismatch at time point %d: Y_test %s vs X_test %s\n', ...
                            t, mat2str(size(Y_t_test)), mat2str(size(X_test)));
                        continue;
                    end
                    
                    % Skip if no variance
                    if std(Y_t_train) < 1e-10 || std(Y_t_test) < 1e-10
                        continue;
                    end
                    
                    % Skip if insufficient data
                    if length(Y_t_train) < 2 || length(Y_t_test) < 1
                        continue;
                    end
                    
                    try
                        % Train model for this time point
                        switch modelType
                            case 'ridge'
                                lambda = 0.1;
                                if length(Y_t_train) > size(X_train, 2) && size(X_train, 2) > 0
                                    % Ensure proper matrix dimensions
                                    XtX = X_train' * X_train; % [nFeatures x nFeatures]
                                    XtY = X_train' * Y_t_train; % [nFeatures x 1]
                                    beta = (XtX + lambda * eye(size(X_train, 2))) \ XtY; % [nFeatures x 1]
                                    Y_pred = X_test * beta; % [nTestTrials x 1]
                                else
                                    Y_pred = repmat(mean(Y_t_train), length(Y_t_test), 1);
                                end
                                
                            case 'lasso'
                                try
                                    if length(Y_t_train) > size(X_train, 2) && size(X_train, 2) > 0
                                        [B, FitInfo] = lasso(X_train, Y_t_train, 'CV', 3, 'NumLambda', 10);
                                        if ~isempty(B) && ~isempty(FitInfo.Intercept)
                                            idxLambda = FitInfo.Index1SE;
                                            intercept = FitInfo.Intercept(idxLambda);
                                            coeffs = B(:, idxLambda); % [nFeatures x 1]
                                            Y_pred = X_test * coeffs + intercept; % [nTestTrials x 1]
                                        else
                                            Y_pred = repmat(mean(Y_t_train), length(Y_t_test), 1);
                                        end
                                    else
                                        Y_pred = repmat(mean(Y_t_train), length(Y_t_test), 1);
                                    end
                                catch ME_lasso
                                    % Fallback to ridge
                                    if size(X_train, 2) > 0
                                        lambda = 0.1;
                                        XtX = X_train' * X_train;
                                        XtY = X_train' * Y_t_train;
                                        beta = (XtX + lambda * eye(size(X_train, 2))) \ XtY;
                                        Y_pred = X_test * beta;
                                    else
                                        Y_pred = repmat(mean(Y_t_train), length(Y_t_test), 1);
                                    end
                                end
                        end
                        
                        % Ensure Y_pred is column vector
                        Y_pred = Y_pred(:);
                        
                        % Final dimension check
                        if length(Y_pred) ~= length(Y_t_test)
                            fprintf('    Prediction dimension mismatch at time point %d: pred %d vs actual %d\n', ...
                                t, length(Y_pred), length(Y_t_test));
                            continue;
                        end
                        
                        % Calculate performance metrics for this time point
                        SS_res = sum((Y_t_test - Y_pred).^2);
                        SS_tot = sum((Y_t_test - mean(Y_t_train)).^2);
                        
                        if SS_tot > 1e-10
                            timePointR2(t, fold) = max(0, 1 - SS_res/SS_tot);
                        else
                            timePointR2(t, fold) = 0;
                        end
                        timePointMSE(t, fold) = mean((Y_t_test - Y_pred).^2);
                        
                        % Store predictions for first fold for visualization
                        if fold == 1
                            if isempty(actualData)
                                actualData = zeros(nTimeBins, length(Y_t_test));
                                predictedData = zeros(nTimeBins, length(Y_t_test));
                            end
                            % Ensure we don't exceed matrix bounds
                            nTestTrialsStored = size(actualData, 2);
                            nTestTrialsCurrent = length(Y_t_test);
                            nTrialsToStore = min(nTestTrialsStored, nTestTrialsCurrent);
                            
                            actualData(t, 1:nTrialsToStore) = Y_t_test(1:nTrialsToStore)';
                            predictedData(t, 1:nTrialsToStore) = Y_pred(1:nTrialsToStore)';
                        end
                        
                    catch ME
                        fprintf('    Error at time point %d: %s\n', t, ME.message);
                        timePointR2(t, fold) = 0;
                        timePointMSE(t, fold) = inf;
                    end
                end
            end
            
            % Average across folds
            avgR2_time = mean(timePointR2, 2); % [nTimeBins x 1]
            avgMSE_time = mean(timePointMSE, 2);
            
            % Store results
            results.(regionName).(featureType).(modelType).R2_timecourse = avgR2_time;
            results.(regionName).(featureType).(modelType).MSE_timecourse = avgMSE_time;
            results.(regionName).(featureType).(modelType).R2_mean = mean(avgR2_time(isfinite(avgR2_time)));
            results.(regionName).(featureType).(modelType).MSE_mean = mean(avgMSE_time(isfinite(avgMSE_time)));
            
            % Store predictions for visualization
            predictionResults.(regionName).(featureType).(modelType).actual = actualData;
            predictionResults.(regionName).(featureType).(modelType).predicted = predictedData;
            
            fprintf('    Average R² across time = %.3f, Average MSE = %.3f\n', ...
                results.(regionName).(featureType).(modelType).R2_mean, ...
                results.(regionName).(featureType).(modelType).MSE_mean);
        end
    end
end

%% ======================== 8. Results Visualization ========================
fprintf('\nGenerating result plots...\n');

% Calculate total number of valid combinations for plotting
validCombinations = {}; % Initialize as cell array
for regionName = fieldnames(results)'
    if ~isempty(fieldnames(results.(regionName{1})))
        for featureType = featureTypes'
            if isfield(results.(regionName{1}), featureType{1})
                for modelType = modelTypes
                    if isfield(results.(regionName{1}).(featureType{1}), modelType{1})
                        validCombinations{end+1, 1} = regionName{1};
                        validCombinations{end, 2} = modelType{1};
                        validCombinations{end, 3} = featureType{1};
                    end
                end
            end
        end
    end
end

if isempty(validCombinations)
    fprintf('No valid model results to plot.\n');
    return;
end

nCombinations = size(validCombinations, 1);
fprintf('Found %d valid model combinations to plot\n', nCombinations);

%% Figure 1: R² Performance Comparison
figure('Position', [50, 100, 800, 600]);
regionList = {};
r2Matrix = [];
legendLabels = {};

for regionName = fieldnames(results)'
    if ~isempty(fieldnames(results.(regionName{1})))
        regionList{end+1} = regionName{1};
        r2Row = [];
        
        for featureType = featureTypes'
            for modelType = modelTypes
                if isfield(results.(regionName{1}), featureType{1}) && ...
                   isfield(results.(regionName{1}).(featureType{1}), modelType{1})
                    r2Row(end+1) = results.(regionName{1}).(featureType{1}).(modelType{1}).R2_mean;
                    if length(regionList) == 1 % Only add labels once
                        legendLabels{end+1} = [upper(modelType{1}(1)), modelType{1}(2:end), ' + ', featureType{1}];
                    end
                else
                    r2Row(end+1) = 0;
                end
            end
        end
        
        r2Matrix = [r2Matrix; r2Row];
    end
end

if ~isempty(r2Matrix) && any(~isnan(r2Matrix(:))) && any(isfinite(r2Matrix(:)))
    bar(r2Matrix);
    xlabel('Brain Region');
    ylabel('Cross-Validated R²');
    title('Model Performance Comparison');
    if ~isempty(legendLabels)
        legend(legendLabels, 'Location', 'best', 'FontSize', 10);
    end
    set(gca, 'XTickLabel', regionList);
    grid on;
    
    % Set ylim only if we have valid data
    validR2 = r2Matrix(~isnan(r2Matrix) & isfinite(r2Matrix));
    if ~isempty(validR2) && max(validR2) > 0
        ylim([0, max(validR2) * 1.1]);
    else
        ylim([0, 0.1]); % Default range if no positive R² values
    end
else
    % Create empty plot with message
    bar([]);
    xlabel('Brain Region');
    ylabel('Cross-Validated R²');
    title('Model Performance Comparison (No Valid Results)');
    grid on;
    ylim([0, 0.1]);
end

saveas(gcf, ['figure/' sesPath '_model_performance_comparison.fig']);
saveas(gcf, ['figure/' sesPath '_model_performance_comparison.png']);

%% Figure 2: Prediction Curves for All Models
nPlots = min(6, nCombinations); % Show up to 6 prediction plots
nRows = ceil(nPlots / 3);
nCols = min(3, nPlots);

figure('Position', [900, 100, 1200, 400*nRows]);

for i = 1:nPlots
    regionName = validCombinations{i, 1};
    modelType = validCombinations{i, 2};
    featureType = validCombinations{i, 3};
    
    if ~isfield(predictionResults, regionName) || ...
       ~isfield(predictionResults.(regionName), featureType) || ...
       ~isfield(predictionResults.(regionName).(featureType), modelType)
        continue;
    end
    
    subplot(nRows, nCols, i);
    
    % Get prediction results for this combination
    actualData = predictionResults.(regionName).(featureType).(modelType).actual;   % [nTimeBins x nTestTrials]
    predictedData = predictionResults.(regionName).(featureType).(modelType).predicted; % [nTimeBins x nTestTrials]
    
    if isempty(actualData) || isempty(predictedData)
        % Show empty plot
        plot([], []);
        xlim([timeStart, timeEnd]);
        xlabel('Time (s)');
        ylabel('Average Firing Rate (Hz)');
        title(sprintf('%s: %s + %s (No Data)', regionName, upper(modelType), featureType), ...
            'FontSize', 11);
        grid on;
        continue;
    end
    
    try
        % Average across test trials to get mean time course
        actualTimeCourse = mean(actualData, 2);   % [nTimeBins x 1]
        predictedTimeCourse = mean(predictedData, 2); % [nTimeBins x 1]
        
        % Calculate overall R² for visualization
        SS_res = sum((actualData(:) - predictedData(:)).^2);
        SS_tot = sum((actualData(:) - mean(actualData(:))).^2);
        overallR2 = max(0, 1 - SS_res/SS_tot);
        
        % Plot actual vs predicted time courses
        plot(time_axis, actualTimeCourse, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Actual');
        hold on;
        plot(time_axis, predictedTimeCourse, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Predicted');
        
        % Add confidence bands (standard error across test trials)
        if size(actualData, 2) > 1
            actualSE = std(actualData, 0, 2) / sqrt(size(actualData, 2));
            predictedSE = std(predictedData, 0, 2) / sqrt(size(predictedData, 2));
            
            % Plot confidence bands
            fill([time_axis, fliplr(time_axis)], ...
                 [actualTimeCourse' - actualSE', fliplr(actualTimeCourse' + actualSE')], ...
                 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            fill([time_axis, fliplr(time_axis)], ...
                 [predictedTimeCourse' - predictedSE', fliplr(predictedTimeCourse' + predictedSE')], ...
                 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
        
        % Add vertical lines for important time points
        xline(0, 'g--', 'Stimulus Onset', 'LineWidth', 1.5, 'FontSize', 8, 'Alpha', 0.7);
        
        xlim([timeStart, timeEnd]);
        xlabel('Time (s)');
        ylabel('Average Firing Rate (Hz)');
        title(sprintf('%s: %s + %s\n(R² = %.3f)', regionName, upper(modelType), featureType, overallR2), ...
            'FontSize', 11);
        legend('Location', 'best', 'FontSize', 9);
        grid on;
        hold off;
        
        % Add text box with additional information
        textStr = sprintf('Trials: %d\nTime bins: %d', size(actualData, 2), size(actualData, 1));
        text(0.02, 0.98, textStr, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'BackgroundColor', 'white', 'EdgeColor', 'gray');
        
    catch ME
        % If prediction visualization fails, show error message
        plot([], []);
        xlim([timeStart, timeEnd]);
        xlabel('Time (s)');
        ylabel('Average Firing Rate (Hz)');
        title(sprintf('%s: %s + %s (Error)', regionName, upper(modelType), featureType), ...
            'FontSize', 11);
        grid on;
        text(0.5, 0.5, sprintf('Error: %s', ME.message), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'Units', 'normalized', 'FontSize', 9);
    end
end

sgtitle(sprintf('Time-Point-wise Neural Activity Predictions - %s', sesPath), 'FontSize', 14);
saveas(gcf, ['figure/' sesPath '_prediction_curves.fig']);
saveas(gcf, ['figure/' sesPath '_prediction_curves.png']);

%% ======================== 9. Results Summary ========================
fprintf('\n==================== 时间点预测结果 ====================\n');

for regionName = fieldnames(results)'
    if ~isempty(fieldnames(results.(regionName{1})))
        fprintf('\n--- %s 脑区 ---\n', regionName{1});
        
        for featureType = featureTypes'
            if isfield(results.(regionName{1}), featureType{1})
                fprintf('  %s 特征:\n', featureType{1});
                
                for modelType = modelTypes
                    if isfield(results.(regionName{1}).(featureType{1}), modelType{1})
                        r2 = results.(regionName{1}).(featureType{1}).(modelType{1}).R2_mean;
                        mse = results.(regionName{1}).(featureType{1}).(modelType{1}).MSE_mean;
                        fprintf('    %s 模型: R² = %.3f, MSE = %.3f\n', modelType{1}, r2, mse);
                    end
                end
            end
        end
    end
end

% Find best performing model
bestR2 = 0;
bestRegion = '';
bestModel = '';
bestFeature = '';

for regionName = fieldnames(results)'
    for featureType = featureTypes'
        for modelType = modelTypes
            if isfield(results.(regionName{1}), featureType{1}) && ...
               isfield(results.(regionName{1}).(featureType{1}), modelType{1})
                r2 = results.(regionName{1}).(featureType{1}).(modelType{1}).R2_mean;
                if r2 > bestR2
                    bestR2 = r2;
                    bestRegion = regionName{1};
                    bestModel = modelType{1};
                    bestFeature = featureType{1};
                end
            end
        end
    end
end

fprintf('\n--- 最佳模型配置 (时间点预测) ---\n');
if bestR2 > 0
    fprintf('最佳配置: %s 脑区, %s 模型, %s 特征 (R² = %.3f)\n', ...
        bestRegion, bestModel, bestFeature, bestR2);
else
    fprintf('没有模型取得正的 R² 值\n');
end

fprintf('\n--- 预测方法 ---\n');
fprintf('目标: 使用行为数据预测每个时间点的神经活动\n');
fprintf('特征: 试验水平的行为变量\n');
fprintf('交叉验证: 基于试验的分割 (在新试验上测试预测性能)\n');
fprintf('时间范围: %.1f 到 %.1f 秒\n', timeStart, timeEnd);

fprintf('\n--- 模型类型说明 ---\n');
fprintf('Ridge: 带有L2正则化的线性回归 (防止过拟合)\n');
fprintf('LASSO: 带有L1正则化的线性回归 (特征选择)\n');

% Save results
save(['postprocessed_data/' sesPath '_regression_results.mat'], 'results', 'featureSets', 'neuralDataByRegion', 'feature_names', 'predictionResults', 'time_axis');

fprintf('\n时间点预测分析完成！结果已保存。\n');

%% ======================== 10. Diagnostic Information ========================
fprintf('\n======================== Diagnostic Information ========================\n');

% Check data properties
fprintf('Behavioral feature matrix properties:\n');
fprintf('  - Shape: %d trials x %d features\n', size(X_standardized));
fprintf('  - Feature variance range: %.3f to %.3f\n', min(var(X_standardized)), max(var(X_standardized)));
fprintf('  - Any NaN values: %s\n', string(any(isnan(X_standardized(:)))));

fprintf('\nBehavioral features used:\n');
for i = 1:length(feature_names)
    fprintf('  - %s: mean = %.3f, std = %.3f\n', feature_names{i}, ...
        mean(X_standardized(:, i)), std(X_standardized(:, i)));
end

fprintf('\nTime-point analysis:\n');
fprintf('  - Time range: %.1f to %.1f seconds\n', timeStart, timeEnd);
fprintf('  - Number of time bins: %d\n', nTimeBins);
fprintf('  - Time resolution: %.3f seconds per bin\n', (timeEnd - timeStart) / nTimeBins);

for regionName = fieldnames(neuralDataByRegion)'
    if isfield(neuralDataByRegion, regionName{1})
        Y_data = neuralDataByRegion.(regionName{1}); % [nNeurons x nTimeBins x nTrials]
        fprintf('\n%s neural data properties:\n', regionName{1});
        fprintf('  - Shape: %d neurons x %d time bins x %d trials\n', size(Y_data));
        fprintf('  - Mean firing rate: %.2f Hz\n', mean(Y_data(:)));
        fprintf('  - Firing rate std: %.2f Hz\n', std(Y_data(:)));
        fprintf('  - Max firing rate: %.2f Hz\n', max(Y_data(:)));
        fprintf('  - Fraction of zero samples: %.3f\n', mean(Y_data(:) == 0));
    end
end 
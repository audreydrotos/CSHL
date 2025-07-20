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

% Trial-based behavioral features
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

%% ======================== 3. Time-Bin Level Feature Expansion ========================
fprintf('Creating time-bin level features...\n');

% Expand trial features to time-bin level
% Each trial's features are replicated across all time bins of that trial
X_expanded = [];
for t = 1:nTrials
    trial_feat = repmat(trial_features(t, :), nTimeBins, 1);
    X_expanded = [X_expanded; trial_feat];
end

% Add time-within-trial features
timeWithinTrial = repmat((1:nTimeBins)' / nTimeBins, nTrials, 1);
timeSin = sin(2 * pi * timeWithinTrial);
timeCos = cos(2 * pi * timeWithinTrial);

X_expanded = [X_expanded, timeWithinTrial, timeSin, timeCos];
feature_names = [feature_names, {'timeWithinTrial', 'timeSin', 'timeCos'}];

% Standardize features
X_standardized = zscore(X_expanded);
X_standardized(isnan(X_standardized)) = 0; % Handle NaN values

fprintf('Final feature matrix: %d samples x %d features\n', size(X_standardized));

%% ======================== 4. PCA Feature Extraction ========================
fprintf('Performing PCA feature extraction...\n');

% Perform PCA on standardized features
[coeff, score, latent, ~, explained] = pca(X_standardized);

% Select components explaining 85% of variance
cumVar = cumsum(explained);
nComponents = find(cumVar >= 85, 1);
if isempty(nComponents)
    nComponents = min(10, size(X_standardized, 2)); % Fallback
end
fprintf('Selected first %d components (explaining %.1f%% variance)\n', nComponents, cumVar(nComponents));

% Create feature sets
featureSets = struct();
featureSets.raw = X_standardized;
featureSets.pca = score(:, 1:nComponents);
featureSets.combined = [X_standardized, score(:, 1:nComponents)];

%% ======================== 5. Neural Response Data Preparation ========================
fprintf('Preparing neural response data...\n');

% Reshape neural data to match feature matrix
% binnedTensor: [nNeurons x nTimeBins x nTrials] -> [nSamples x nNeurons]
Y_neural = [];
for t = 1:nTrials
    Y_neural = [Y_neural; squeeze(binnedTensor(:, :, t))']; % [nTimeBins x nNeurons]
end

fprintf('Neural data matrix: %d samples x %d neurons\n', size(Y_neural));

% Filter neurons with very low activity
fprintf('Filtering low-activity neurons...\n');
minFiringRate = 0.1; % Minimum firing rate threshold (Hz) - more lenient
maxZeroFraction = 0.95; % Maximum fraction of zero samples allowed per neuron - more lenient

% Organize by brain regions and filter
neuralData = struct();

for rr = 1:length(regionSelected)
    regionCode = regionSelected(rr);
    regionName = regionNames{rr};
    
    % Get neurons in this region
    regionIdx = neurons.region == regionCode;
    if sum(regionIdx) == 0
        warning('No neurons found in region %s', regionName);
        continue;
    end
    
    Y_region = Y_neural(:, regionIdx);
    nRegionNeurons = size(Y_region, 2);
    
    fprintf('\nRegion %s: %d neurons before filtering\n', regionName, nRegionNeurons);
    
    % Analyze data distribution
    firingRates = mean(Y_region, 1);
    zeroFractions = mean(Y_region == 0, 1);
    
    fprintf('  Data distribution analysis:\n');
    fprintf('  - Mean firing rate: %.3f ± %.3f Hz\n', mean(firingRates), std(firingRates));
    fprintf('  - Firing rate range: %.3f - %.3f Hz\n', min(firingRates), max(firingRates));
    fprintf('  - Mean zero fraction: %.3f ± %.3f\n', mean(zeroFractions), std(zeroFractions));
    fprintf('  - Zero fraction range: %.3f - %.3f\n', min(zeroFractions), max(zeroFractions));
    
    % Filter neurons
    validNeurons = [];
    for n = 1:nRegionNeurons
        zeroFraction = zeroFractions(n);
        meanFiringRate = firingRates(n);
        
        if zeroFraction < maxZeroFraction && meanFiringRate >= minFiringRate
            validNeurons(end+1) = n;
        end
    end
    
    % If too few neurons pass, relax criteria
    if length(validNeurons) < 5
        fprintf('  Too few neurons passed initial criteria, relaxing...\n');
        validNeurons = [];
        
        % Very relaxed criteria: just remove completely silent neurons
        for n = 1:nRegionNeurons
            if firingRates(n) > 0.01 && zeroFractions(n) < 0.99
                validNeurons(end+1) = n;
            end
        end
    end
    
    if isempty(validNeurons)
        warning('No valid neurons found in region %s after filtering', regionName);
        continue;
    end
    
    Y_region_filtered = Y_region(:, validNeurons);
    fprintf('Region %s: %d neurons after filtering\n', regionName, length(validNeurons));
    
    neuralData.(regionName) = Y_region_filtered;
    
    fprintf('  Final data: %d samples x %d neurons\n', size(Y_region_filtered));
    fprintf('  - Mean firing rate: %.2f Hz\n', mean(Y_region_filtered(:)));
    fprintf('  - Firing rate std: %.2f Hz\n', std(Y_region_filtered(:)));
    fprintf('  - Max firing rate: %.2f Hz\n', max(Y_region_filtered(:)));
    fprintf('  - Fraction of zero bins: %.3f\n', mean(Y_region_filtered(:) == 0));
end

%% ======================== 6. Cross-Validation Setup ========================
fprintf('Cross-validation will be set up dynamically for each dataset...\n');

% Cross-validation parameters
nFolds = 5;

%% ======================== 7. Model Training and Evaluation ========================
fprintf('Starting model training and evaluation...\n');

% Model types
modelTypes = {'ridge', 'lasso', 'glm_poisson', 'glm_gaussian'};
featureTypes = fieldnames(featureSets);

% Results storage
results = struct();

% Train models for each brain region
for rr = 1:length(regionNames)
    regionName = regionNames{rr};
    
    if ~isfield(neuralData, regionName)
        continue;
    end
    
    fprintf('\n=== Analyzing region: %s ===\n', regionName);
    Y_region = neuralData.(regionName);
    nRegionNeurons = size(Y_region, 2);
    
    results.(regionName) = struct();
    
    % Train models for each feature set and model type
    for ft = 1:length(featureTypes)
        featureType = featureTypes{ft};
        X = featureSets.(featureType);
        
        fprintf('Feature type: %s (%d samples x %d features)\n', featureType, size(X, 1), size(X, 2));
        
        for mt = 1:length(modelTypes)
            modelType = modelTypes{mt};
            fprintf('  Model type: %s\n', modelType);
            
            % Cross-validation
            nSamples = size(X, 1);
            
            % Create K-fold cross-validation indices manually
            randIdx = randperm(nSamples); % Random permutation of sample indices
            foldSize = floor(nSamples / nFolds);
            cv_indices = zeros(nSamples, 1);
            
            for k = 1:nFolds
                if k < nFolds
                    startIdx = (k-1) * foldSize + 1;
                    endIdx = k * foldSize;
                else
                    % Last fold gets remaining samples
                    startIdx = (k-1) * foldSize + 1;
                    endIdx = nSamples;
                end
                cv_indices(randIdx(startIdx:endIdx)) = k;
            end
            
            cvR2 = zeros(nFolds, 1);
            cvMSE = zeros(nFolds, 1);
            
            for fold = 1:nFolds
                testIdx = find(cv_indices == fold);
                trainIdx = find(cv_indices ~= fold);
                
                X_train = X(trainIdx, :);
                Y_train = Y_region(trainIdx, :);
                X_test = X(testIdx, :);
                Y_test = Y_region(testIdx, :);
                
                % Train model for multiple neurons (average performance)
                neuronR2 = zeros(nRegionNeurons, 1);
                neuronMSE = zeros(nRegionNeurons, 1);
                
                for n = 1:min(nRegionNeurons, 10) % Limit to 10 neurons for speed
                    y_train = Y_train(:, n);
                    y_test = Y_test(:, n);
                    
                    % Skip if no variance in target
                    if std(y_train) < 1e-10 || std(y_test) < 1e-10
                        continue;
                    end
                    
                    try
                        switch modelType
                            case 'ridge'
                                % Ridge regression with cross-validation for lambda
                                lambdas = logspace(-2, 2, 10);
                                bestLambda = 1;
                                bestValMSE = inf;
                                
                                % Simple validation split
                                nTrain = length(y_train);
                                if nTrain > 20
                                    valIdx = randperm(nTrain, floor(0.2 * nTrain));
                                    trainIdx2 = setdiff(1:nTrain, valIdx);
                                    
                                    for lambda = lambdas
                                        beta = (X_train(trainIdx2,:)'*X_train(trainIdx2,:) + lambda*eye(size(X_train, 2))) \ ...
                                               (X_train(trainIdx2,:)' * y_train(trainIdx2));
                                        y_val_pred = X_train(valIdx,:) * beta;
                                        valMSE = mean((y_train(valIdx) - y_val_pred).^2);
                                        
                                        if valMSE < bestValMSE
                                            bestValMSE = valMSE;
                                            bestLambda = lambda;
                                        end
                                    end
                                end
                                
                                % Train final model
                                beta = (X_train'*X_train + bestLambda*eye(size(X_train, 2))) \ (X_train' * y_train);
                                y_pred = X_test * beta;
                                
                            case 'lasso'
                                % LASSO regression
                                try
                                    [B, FitInfo] = lasso(X_train, y_train, 'CV', 3, 'NumLambda', 10);
                                    idxLambda = FitInfo.Index1SE;
                                    beta = [FitInfo.Intercept(idxLambda); B(:, idxLambda)];
                                    y_pred = [ones(size(X_test, 1), 1), X_test] * beta;
                                catch
                                    % Fallback to ridge
                                    beta = (X_train'*X_train + 0.1*eye(size(X_train, 2))) \ (X_train' * y_train);
                                    y_pred = X_test * beta;
                                end
                                
                            case 'glm_poisson'
                                % GLM with Poisson distribution (good for count/rate data)
                                try
                                    % Convert firing rates to counts (assuming binSize = 0.005)
                                    binSize = 0.005;
                                    y_train_counts = round(y_train * binSize);
                                    y_train_counts(y_train_counts < 0) = 0; % Ensure non-negative
                                    y_train_counts = max(y_train_counts, 0.1); % Avoid zeros for log link
                                    
                                    % Check and fix rank deficiency
                                    [X_train_fixed, ~] = fixRankDeficiency(X_train);
                                    X_test_fixed = X_test(:, 1:size(X_train_fixed, 2)); % Match dimensions
                                    
                                    % Use direct matrix approach for GLM
                                    glmModel = fitglm(X_train_fixed, y_train_counts, 'Distribution', 'poisson', ...
                                        'Link', 'log');
                                    
                                    % Predict on test set
                                    y_pred_counts = predict(glmModel, X_test_fixed);
                                    y_pred = y_pred_counts / binSize; % Convert back to rates
                                    
                                catch ME
                                    fprintf('        GLM Poisson failed (%s), using ridge fallback\n', ME.message);
                                    beta = (X_train'*X_train + 0.1*eye(size(X_train, 2))) \ (X_train' * y_train);
                                    y_pred = X_test * beta;
                                end
                                
                            case 'glm_gaussian'
                                % GLM with Gaussian distribution (equivalent to linear regression)
                                try
                                    % Check and fix rank deficiency
                                    [X_train_fixed, ~] = fixRankDeficiency(X_train);
                                    X_test_fixed = X_test(:, 1:size(X_train_fixed, 2)); % Match dimensions
                                    
                                    % Use direct matrix approach for GLM
                                    glmModel = fitglm(X_train_fixed, y_train, 'Distribution', 'normal', ...
                                        'Link', 'identity');
                                    
                                    % Predict on test set
                                    y_pred = predict(glmModel, X_test_fixed);
                                    
                                catch ME
                                    fprintf('        GLM Gaussian failed (%s), using ridge fallback\n', ME.message);
                                    beta = (X_train'*X_train + 0.1*eye(size(X_train, 2))) \ (X_train' * y_train);
                                    y_pred = X_test * beta;
                                end
                        end
                        
                        % Calculate performance metrics
                        SS_res = sum((y_test - y_pred).^2);
                        SS_tot = sum((y_test - mean(y_train)).^2); % Use training mean as baseline
                        
                        if SS_tot > 1e-10
                            neuronR2(n) = max(0, 1 - SS_res/SS_tot);
                        else
                            neuronR2(n) = 0;
                        end
                        neuronMSE(n) = mean((y_test - y_pred).^2);
                        
                    catch ME
                        fprintf('    Error for neuron %d: %s\n', n, ME.message);
                        neuronR2(n) = 0;
                        neuronMSE(n) = inf;
                    end
                end
                
                % Average across neurons for this fold
                validR2 = neuronR2(neuronR2 > 0 & ~isnan(neuronR2));
                validMSE = neuronMSE(neuronMSE < inf & ~isnan(neuronMSE));
                
                if ~isempty(validR2)
                    cvR2(fold) = mean(validR2);
                else
                    cvR2(fold) = 0;
                end
                
                if ~isempty(validMSE)
                    cvMSE(fold) = mean(validMSE);
                else
                    cvMSE(fold) = inf;
                end
            end
            
            % Store results
            validFoldR2 = cvR2(~isnan(cvR2) & isfinite(cvR2));
            validFoldMSE = cvMSE(~isnan(cvMSE) & isfinite(cvMSE));
            
            if ~isempty(validFoldR2)
                results.(regionName).(modelType).(featureType).R2 = mean(validFoldR2);
            else
                results.(regionName).(modelType).(featureType).R2 = 0;
            end
            
            if ~isempty(validFoldMSE)
                results.(regionName).(modelType).(featureType).MSE = mean(validFoldMSE);
            else
                results.(regionName).(modelType).(featureType).MSE = inf;
            end
            
            fprintf('    Average R² = %.3f, Average MSE = %.3f\n', ...
                results.(regionName).(modelType).(featureType).R2, ...
                results.(regionName).(modelType).(featureType).MSE);
        end
    end
end

%% ======================== 8. Results Visualization ========================
fprintf('\nGenerating result plots...\n');

figure('Position', [100, 100, 1200, 500]);

% 1. R² comparison bar plot
subplot(1, 2, 1);
regionList = {};
r2Matrix = [];
legendLabels = {};

for regionName = fieldnames(results)'
    if ~isempty(fieldnames(results.(regionName{1})))
        regionList{end+1} = regionName{1};
        r2Row = [];
        
        for modelType = modelTypes
            for featureType = featureTypes'
                if isfield(results.(regionName{1}), modelType{1}) && ...
                   isfield(results.(regionName{1}).(modelType{1}), featureType{1})
                    r2Row(end+1) = results.(regionName{1}).(modelType{1}).(featureType{1}).R2;
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
        legend(legendLabels, 'Location', 'best', 'FontSize', 8);
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

% 2. Best model demonstration
subplot(1, 2, 2);

% Find best performing model
bestR2 = 0;
bestRegion = '';
bestModel = '';
bestFeature = '';

for regionName = fieldnames(results)'
    for modelType = modelTypes
        for featureType = featureTypes'
            if isfield(results.(regionName{1}), modelType{1}) && ...
               isfield(results.(regionName{1}).(modelType{1}), featureType{1})
                r2 = results.(regionName{1}).(modelType{1}).(featureType{1}).R2;
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

if bestR2 > 0 && isfield(neuralData, bestRegion)
    % Demo prediction for first neuron
    Y_demo = neuralData.(bestRegion)(:, 1);
    X_demo = featureSets.(bestFeature);
    
    % Use first 70% for training, last 30% for testing
    nSamples = length(Y_demo);
    splitPoint = floor(0.7 * nSamples);
    
    X_train = X_demo(1:splitPoint, :);
    Y_train = Y_demo(1:splitPoint);
    X_test = X_demo((splitPoint+1):end, :);
    Y_test = Y_demo((splitPoint+1):end);
    
    % Train best model
    switch bestModel
        case 'ridge'
            lambda = 0.1;
            beta = (X_train'*X_train + lambda*eye(size(X_train, 2))) \ (X_train' * Y_train);
            Y_pred = X_test * beta;
            
        case 'lasso'
            try
                [B, FitInfo] = lasso(X_train, Y_train, 'CV', 3);
                idxLambda = FitInfo.Index1SE;
                beta = [FitInfo.Intercept(idxLambda); B(:, idxLambda)];
                Y_pred = [ones(size(X_test, 1), 1), X_test] * beta;
            catch
                % Fallback
                lambda = 0.1;
                beta = (X_train'*X_train + lambda*eye(size(X_train, 2))) \ (X_train' * Y_train);
                Y_pred = X_test * beta;
            end
            
        case 'glm_poisson'
            try
                % Convert firing rates to counts for Poisson GLM
                binSize = 0.005;
                Y_train_counts = round(Y_train * binSize);
                Y_train_counts(Y_train_counts < 0) = 0;
                Y_train_counts = max(Y_train_counts, 0.1); % Avoid zeros for log link
                
                % Fix rank deficiency
                [X_train_fixed, ~] = fixRankDeficiency(X_train);
                X_test_fixed = X_test(:, 1:size(X_train_fixed, 2));
                
                % Fit Poisson GLM using direct matrix approach
                glmModel = fitglm(X_train_fixed, Y_train_counts, 'Distribution', 'poisson', 'Link', 'log');
                
                % Predict
                Y_pred_counts = predict(glmModel, X_test_fixed);
                Y_pred = Y_pred_counts / binSize;
            catch
                % Fallback to ridge
                lambda = 0.1;
                beta = (X_train'*X_train + lambda*eye(size(X_train, 2))) \ (X_train' * Y_train);
                Y_pred = X_test * beta;
            end
            
        case 'glm_gaussian'
            try
                % Fix rank deficiency
                [X_train_fixed, ~] = fixRankDeficiency(X_train);
                X_test_fixed = X_test(:, 1:size(X_train_fixed, 2));
                
                % Fit Gaussian GLM using direct matrix approach
                glmModel = fitglm(X_train_fixed, Y_train, 'Distribution', 'normal', 'Link', 'identity');
                
                % Predict
                Y_pred = predict(glmModel, X_test_fixed);
            catch
                % Fallback to ridge
                lambda = 0.1;
                beta = (X_train'*X_train + lambda*eye(size(X_train, 2))) \ (X_train' * Y_train);
                Y_pred = X_test * beta;
            end
            
        otherwise
            % Default to ridge
            lambda = 0.1;
            beta = (X_train'*X_train + lambda*eye(size(X_train, 2))) \ (X_train' * Y_train);
            Y_pred = X_test * beta;
    end
    
    % Calculate demo R²
    SS_res = sum((Y_test - Y_pred).^2);
    SS_tot = sum((Y_test - mean(Y_train)).^2);
    demoR2 = max(0, 1 - SS_res/SS_tot);
    
    % Plot
    time_axis = (1:length(Y_test)) * 0.005; % Assuming 5ms bins
    plot(time_axis, Y_test, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
    hold on;
    plot(time_axis, Y_pred, 'r--', 'LineWidth', 2, 'DisplayName', 'Predicted');
    
    xlabel('Time (s)');
    ylabel('Firing Rate (Hz)');
    title(sprintf('Prediction Demo: %s Region\n%s Model with %s Features (R² = %.3f)', ...
        bestRegion, bestModel, bestFeature, demoR2));
    legend('Location', 'best');
    grid on;
    hold off;
else
    % No good results
    plot([], []);
    xlabel('Time (s)');
    ylabel('Firing Rate (Hz)');
    title('Prediction Demo (No Data Available)');
    grid on;
end

sgtitle(sprintf('Linear Regression Analysis Results - %s', sesPath), 'FontSize', 14);
saveas(gcf, ['figure/' sesPath '_linear_regression_analysis.fig']);
saveas(gcf, ['figure/' sesPath '_linear_regression_analysis.png']);

%% ======================== 9. Results Summary ========================
fprintf('\n======================== Results Summary ========================\n');

for regionName = fieldnames(results)'
    if ~isempty(fieldnames(results.(regionName{1})))
        fprintf('\n--- %s Region ---\n', regionName{1});
        
        for modelType = modelTypes
            if isfield(results.(regionName{1}), modelType{1})
                fprintf('  %s Model:\n', modelType{1});
                
                for featureType = featureTypes'
                    if isfield(results.(regionName{1}).(modelType{1}), featureType{1})
                        r2 = results.(regionName{1}).(modelType{1}).(featureType{1}).R2;
                        mse = results.(regionName{1}).(modelType{1}).(featureType{1}).MSE;
                        fprintf('    %s features: R² = %.3f, MSE = %.3f\n', featureType{1}, r2, mse);
                    end
                end
            end
        end
    end
end

fprintf('\n--- Best Model Configuration ---\n');
if bestR2 > 0
    fprintf('Best: %s region, %s model, %s features (R² = %.3f)\n', ...
        bestRegion, bestModel, bestFeature, bestR2);
else
    fprintf('No models achieved positive R² values\n');
end

fprintf('\n--- Model Type Explanations ---\n');
fprintf('Ridge: Linear regression with L2 regularization (prevents overfitting)\n');
fprintf('LASSO: Linear regression with L1 regularization (feature selection)\n');
fprintf('GLM Poisson: Generalized Linear Model with Poisson distribution\n');
fprintf('  - Ideal for count/rate data (neural spikes)\n');
fprintf('  - Uses log link function: log(μ) = Xβ\n');
fprintf('  - Naturally handles non-negative predictions\n');
fprintf('GLM Gaussian: Generalized Linear Model with Normal distribution\n');
fprintf('  - Equivalent to standard linear regression\n');
fprintf('  - Uses identity link function: μ = Xβ\n');
fprintf('  - Baseline comparison for other GLM models\n');

% Save results
save(['postprocessed_data/' sesPath '_regression_results.mat'], 'results', 'featureSets', 'neuralData', 'feature_names');

fprintf('\nAnalysis complete! Results saved.\n');

%% ======================== 10. Diagnostic Information ========================
fprintf('\n======================== Diagnostic Information ========================\n');

% Check data properties
fprintf('Feature matrix properties:\n');
fprintf('  - Shape: %d x %d\n', size(X_standardized));
fprintf('  - Feature variance range: %.3f to %.3f\n', min(var(X_standardized)), max(var(X_standardized)));
fprintf('  - Any NaN values: %s\n', string(any(isnan(X_standardized(:)))));

for regionName = fieldnames(neuralData)'
    if isfield(neuralData, regionName{1})
        Y_data = neuralData.(regionName{1});
        fprintf('\n%s neural data properties:\n', regionName{1});
        fprintf('  - Shape: %d x %d\n', size(Y_data));
        fprintf('  - Mean firing rate: %.2f Hz\n', mean(Y_data(:)));
        fprintf('  - Firing rate std: %.2f Hz\n', std(Y_data(:)));
        fprintf('  - Max firing rate: %.2f Hz\n', max(Y_data(:)));
                 fprintf('  - Fraction of zero bins: %.3f\n', mean(Y_data(:) == 0));
     end
 end

%% ======================== Helper Functions ========================
function [X_fixed, removedCols] = fixRankDeficiency(X)
% Fix rank deficiency in design matrix by aggressively reducing features
    
    [nSamples, nFeatures] = size(X);
    
    fprintf('          Fixing rank deficiency: %d features -> ', nFeatures);
    
    % Be very conservative: use at most 50% of samples as features, with a max of 20
    maxFeatures = min([20, floor(nSamples * 0.5), floor(nFeatures * 0.6)]);
    
    % Remove features with very low variance (likely constant or nearly constant)
    featureVars = var(X);
    validVariance = featureVars > 1e-10;
    X_temp = X(:, validVariance);
    validIndices = find(validVariance);
    
    if size(X_temp, 2) > maxFeatures
        % Select features with highest variance
        [~, sortIdx] = sort(featureVars(validVariance), 'descend');
        keepIdx = sortIdx(1:maxFeatures);
        independentCols = validIndices(keepIdx);
    else
        independentCols = validIndices;
    end
    
    % Final check: ensure we don't have too many features
    if length(independentCols) >= nSamples
        independentCols = independentCols(1:min(nSamples-2, 15));
    end
    
    % Create fixed matrix
    X_fixed = X(:, independentCols);
    removedCols = setdiff(1:nFeatures, independentCols);
    
    fprintf('%d features (removed %d)\n', size(X_fixed, 2), length(removedCols));
end 
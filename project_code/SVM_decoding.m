% load tensor data
addpath(genpath('..'))

close all; clear; clc; rng(123);
%% pick a session
sesPath = 'Moniz_2017-05-16';
%sesPath = 'Forssmann_2017-11-01';
%sesPath = 'Lederberg_2017-12-05'; 
load(['postprocessed_data/' sesPath '_binnedTensor.mat'])
load(['postprocessed_data/' sesPath '_S.mat'])
load(['postprocessed_data/' sesPath '_regions.mat'])
load(['postprocessed_data/' sesPath '_neurons.mat'])
load(['postprocessed_data/' sesPath '_trials.mat'])
 
%% fit linear regression model
leftStim = S.trials.visualStim_contrastLeft;
rightStim = S.trials.visualStim_contrastRight;
% 
% region_code = 3; % LGd
% region_idx = neurons.region == region_code;
% region_neurons = binnedTensor(region_idx, :, :);
% 
% % predict the right contrast stimulus from the data
% % for each individual neuron
% nNeurons = size(region_neurons,1);
% 
% accuracy_matrix = zeros(1,nNeurons);
% accuracy_matrix_shuffled = zeros(1,nNeurons);
% 
% % use SVM
% parfor i = 1:nNeurons
%     % what do we want to predict with SVM
%     y = rightStim;
% 
%     % which data to use
%     neuron_data = region_neurons(i,:,:);
%     neuron_data = squeeze(neuron_data);
%     neuron_data = neuron_data'; % flip so this is rows = trials and columns = binned spike times
%     x = neuron_data;
% 
%     % set up cross validation
%     cv = cvpartition(size(neuron_data, 1), 'HoldOut', 0.3); % 70% training, 30% testing
%     trainIdx = training(cv);
%     testIdx = test(cv);
% 
%     X_train = x(trainIdx, :);
%     y_train = y(trainIdx);
%     X_test = x(testIdx, :);
%     y_test = y(testIdx);
% 
%     % Train SVM model
%     svmModel = fitcecoc(X_train, y_train); % fits SVM with default linear kernel, one v one
% 
%     % Test the model
%     predictions = predict(svmModel, X_test);
% 
%     % Evaluate performance
%     accuracy = sum(predictions == y_test) / length(y_test) * 100;
%     accuracy_matrix(i) = accuracy;
% 
%     % next, need to shuffle the labels and compare to shuffled accuracy
%     yshuffled = y(randperm(length(y)))
% 
%     X_train = x(trainIdx, :);
%     y_train = yshuffled(trainIdx);
%     X_test = x(testIdx, :);
%     y_test = yshuffled(testIdx);
% 
%     % Train SVM model
%     svmModel = fitcecoc(X_train, y_train); % fits SVM with default linear kernel, one v one
% 
%     % Test the model
%     predictions = predict(svmModel, X_test);
% 
%     % Evaluate performance
%     accuracy = sum(predictions == y_test) / length(y_test) * 100;
%     accuracy_matrix_shuffled(i) = accuracy;
% 
% end
% 
% figure()
% histogram(accuracy_matrix)
% hold on
% histogram(accuracy_matrix_shuffled)
% xline(length(unique(rightStim)))

%% predict right stimulus across each window of the binned spike data
% from neural activity predict across TIME to see when prediction is
% highest for the sound. predict once for every time window.

% load tensor data and smooth it
load(['postprocessed_data/' sesPath '_binnedTensor.mat'])
smoothedTensor = movmean(binnedTensor, [5 5], 2);
stepSize = 10;
idx = 1:stepSize:size(smoothedTensor,2);
tensorPCA = smoothedTensor(:,idx,:);

%%
% set up bins needed on X axis
nBins = size(tensorPCA, 2);
binnedTime = 2.5;
msBin = binnedTime/nBins;
time = -0.5:msBin:2;
% bin tick every 50 bins (10 values)
idx = 1:10:51;
time_xaxis = time(idx);

% regions
regions = [3 10 8];

% 5 fold cross validation
cv = cvpartition(y,'KFold', 5);

% What do we want to decode? 
% y = rightStim;
% y = trials.Correct;
y = S.trials.response_choice;

% initialize accuracy
accuracy = zeros(size(region_neurons,2), length(region_code));

for r = 1:length(regions)
    region_code = regions(r); % LGd
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);

    % perform decoding across time windows
    for w = 1:size(region_neurons,2)
        
        % go across each time window
        x_window = squeeze(region_neurons(:,w,:));
        x = x_window';
    
        % initialize matrix
        acc = zeros(cv.NumTestSets,1);
        allScores = [];
        allLabels = [];

        % train and test the model
        for i = 1:cv.NumTestSets
            trainX = x(cv.training(i),:);
            testX = x(cv.test(i),:);
            trainY = y(cv.training(i));
            testY = y(cv.test(i));
    
            model = fitcecoc(trainX, trainY);
            [pred, score] = predict(model, testX);
            acc(i) = mean(pred == testY);
        end
        accuracy(w,r) = mean(acc) * 100;
    end
end

figure('Position', [0 0 300 800])
subplot(3, 1, 1)
plot(accuracy(:,1))
hold on
ylim([0 100])
ylabel('Accuracy')
xlabel('Time relative to Stim. Onset')
xticks(idx)
xticklabels(time_xaxis)
title({'Decoding Actual Choice', 'from LGD activity using SVM'})
hold off

subplot(3, 1, 2)
plot(accuracy(:,2))
title({'Decoding Actual Choice', 'from VIsp activity using SVM'})
ylabel('Accuracy')
xlabel('Time relative to Stim. Onset')
xticks(idx)
ylim([0 100])
xticklabels(time_xaxis)

subplot(3, 1, 3)
plot(accuracy(:,3))
title({'Decoding Actual Choice', 'from TH activity using SVM'})
ylabel('Accuracy')
xlabel('Time relative to Stim. Onset')
xticks(idx)
ylim([0 100])
xticklabels(time_xaxis)

savefig('actual_choice_decoding.fig')
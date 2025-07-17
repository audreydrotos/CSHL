% load tensor data
load('binnedTensor.mat')

%% fit linear regression model
leftStim = S.trials.visualStim_contrastLeft;
rightStim = S.trials.visualStim_contrastRight;

region_code = 3; % LGd
region_idx = neurons.region == region_code;
region_neurons = binnedTensor(region_idx, :, :);

% predict the right contrast stimulus from the data
% for each individual neuron
nNeurons = size(region_neurons,1);

accuracy_matrix = zeros(1,nNeurons);
accuracy_matrix_shuffled = zeros(1,nNeurons);

% use SVM
parfor i = 1:nNeurons
    % what do we want to predict with SVM
    y = rightStim;

    % which data to use
    neuron_data = region_neurons(i,:,:);
    neuron_data = squeeze(neuron_data);
    neuron_data = neuron_data'; % flip so this is rows = trials and columns = binned spike times
    x = neuron_data;

    % set up cross validation
    cv = cvpartition(size(neuron_data, 1), 'HoldOut', 0.3); % 70% training, 30% testing
    trainIdx = training(cv);
    testIdx = test(cv);
    
    X_train = x(trainIdx, :);
    y_train = y(trainIdx);
    X_test = x(testIdx, :);
    y_test = y(testIdx);
    
    % Train SVM model
    svmModel = fitcecoc(X_train, y_train); % fits SVM with default linear kernel, one v one
    
    % Test the model
    predictions = predict(svmModel, X_test);
    
    % Evaluate performance
    accuracy = sum(predictions == y_test) / length(y_test) * 100;
    accuracy_matrix(i) = accuracy;

    % next, need to shuffle the labels and compare to shuffled accuracy
    yshuffled = y(randperm(length(y)))

    X_train = x(trainIdx, :);
    y_train = shuffledy(trainIdx);
    X_test = x(testIdx, :);
    y_test = shuffledy(testIdx);

    % Train SVM model
    svmModel = fitcecoc(X_train, y_train); % fits SVM with default linear kernel, one v one
    
    % Test the model
    predictions = predict(svmModel, X_test);
    
    % Evaluate performance
    accuracy = sum(predictions == y_test) / length(y_test) * 100;
    accuracy_matrix_shuffled(i) = accuracy;

end

figure()
histogram(accuracy_matrix)
hold on
histogram(accuracy_matrix_shuffled)
xline(length(unique(rightStim)))

%% predict right stimulus across each window of the binned spike data
% from neural activity predict across TIME to see when prediction is
% highest for the sound. predict once for every time window.

% load tensor data and smooth it
load('binnedTensor.mat')
smoothedTensor = movmean(binnedTensor, [5 5], 2);
stepSize = 10;
idx = 1:stepSize:size(smoothedTensor,2);
tensorPCA = smoothedTensor(:,idx,:);

% subset for our neurons
region_code = 3; % LGd
region_idx = neurons.region == region_code;
region_neurons = tensorPCA(region_idx, :, :);

% what do we want to predict with SVM
y = rightStim;

% initialize matrix
accuracy = [];

% 5 fold cross validation
cv = cvpartition(y,'KFold', 5);

% perform decoding across time windows
parfor w = 1:size(neuron_data,2)
    x_window = squeeze(region_neurons(:,w,:));
    x = x_window';

    % initialize matrix
    acc = zeros(cv.NumTestSets,1);

    % train and test the model
    for i = 1:cv.NumTestSets
        trainX = x(cv.training(i),:);
        testX = x(cv.test(i),:);
        trainY = y(cv.training(i));
        testY = y(cv.test(i));

        model = fitcecoc(trainX, trainY);
        pred = predict(model, testX);
        acc(i) = mean(pred == testY);
    end

    accuracy(w) = mean(acc) * 100;
end

plot(accuracy)
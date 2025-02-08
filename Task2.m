clc; clear; close all;

% Define the folder containing the .mat files
data_folder = '/Users/nisayurusandaneth/Downloads/CW-Data 2/CW'; 

% Get all .mat file names in the folder
files = dir(fullfile(data_folder, '*.mat'));

% Check if the folder is empty
if isempty(files)
    error('No .mat files found in the specified folder.');
end

% Initialize structures to store features by group
feature_groups = struct();

% Load the data and group by feature count
for i = 1:length(files)
    file_path = fullfile(data_folder, files(i).name);
    data = load(file_path);
    
    variable_names = fieldnames(data);
    features = data.(variable_names{1});
    
    num_features = size(features, 2);
    group_name = sprintf('F%d_features', num_features);
    if ~isfield(feature_groups, group_name)
        feature_groups.(group_name) = [];
    end
    feature_groups.(group_name) = [feature_groups.(group_name); features];
end

% Train MLP for each feature group and evaluate
mlp_models = struct();
performances = struct();

feature_group_names = fieldnames(feature_groups);
for i = 1:length(feature_group_names)
    group_name = feature_group_names{i};
    features = feature_groups.(group_name);

    % Define placeholder labels (for binary classification example)
    labels = mod(1:size(features, 1), 2); % Alternate 0 and 1 for binary labels

    % Split dataset into training, validation, and testing sets
    num_samples = size(features, 1);
    indices = randperm(num_samples);
    train_idx = indices(1:round(0.7 * num_samples));
    val_idx = indices(round(0.7 * num_samples) + 1:round(0.85 * num_samples));
    test_idx = indices(round(0.85 * num_samples) + 1:end);

    X_train = features(train_idx, :);
    Y_train = labels(train_idx)';
    X_val = features(val_idx, :);
    Y_val = labels(val_idx)';
    X_test = features(test_idx, :);
    Y_test = labels(test_idx)';

    % Normalize features
    mu = mean(X_train);
    sigma = std(X_train);
    X_train = (X_train - mu) ./ sigma;
    X_val = (X_val - mu) ./ sigma;
    X_test = (X_test - mu) ./ sigma;

    % Hyperparameter Tuning for Hidden Layers
    hidden_layer_sizes = [10, 20, 30, 40, 50]; % Added more sizes
    best_model = [];
    best_val_accuracy = 0;

    for h = hidden_layer_sizes
        net = patternnet(h, 'trainlm');
        net.performParam.regularization = 0.01; % Adjusted L2 Regularization
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;

        % Train the network
        [net, tr] = train(net, X_train', Y_train');

        % Evaluate on validation set
        Y_val_pred = net(X_val');
        val_accuracy = sum(round(Y_val_pred') == Y_val) / length(Y_val);

        if val_accuracy > best_val_accuracy
            best_val_accuracy = val_accuracy;
            best_model = net;
        end
    end

    % Evaluate the best model on the test set
    Y_test_pred = best_model(X_test');
    test_accuracy = sum(round(Y_test_pred') == Y_test) / length(Y_test);

    % Store the results
    mlp_models.(group_name) = best_model;
    performances.(group_name) = struct('val_accuracy', best_val_accuracy, 'test_accuracy', test_accuracy);

    fprintf('Group: %s, Validation Accuracy: %.4f, Test Accuracy: %.4f\n', ...
        group_name, best_val_accuracy, test_accuracy);
end

% Plot test performance comparison across feature groups
group_names = fieldnames(performances);
test_accuracies = zeros(1, length(group_names));

for i = 1:length(group_names)
    group_name = group_names{i};
    test_accur
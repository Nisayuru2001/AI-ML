clc; clear; close all;

% Define the folder containing the .mat files
data_folder = '/Users/nisayurusandaneth/Downloads/CW-Data 2';

% Get all .mat file names in the folder
files = dir(fullfile(data_folder, '*.mat'));

% Check if the folder is empty
if isempty(files)
    error('No .mat files found in the specified folder.');
end

% Initialize structures to store statistics
user_statistics = struct();
grouped_files = struct();

% Loop through each .mat file
for i = 1:length(files)
    % Skip the 'processed_data.mat' file
    if strcmp(files(i).name, 'processed_data.mat')
        continue; % Skip this file
    end

    % Load the file
    file_path = fullfile(data_folder, files(i).name);
    data = load(file_path);
    
    % Extract the variable (assuming the first variable in the file)
    variable_names = fieldnames(data);
    features = data.(variable_names{1});
    
    % Ensure features are numeric and valid
    if ~isnumeric(features)
        if iscell(features)
            try
                features = cell2mat(features);  % Convert cell array to numeric matrix
            catch
                warning('Skipping file %s because the features are not numeric or could not be converted.', files(i).name);
                continue;
            end
        else
            warning('Skipping file %s because the features are not numeric.', files(i).name);
            continue;
        end
    end

    % Calculate descriptive statistics
    mean_features = mean(features, 1);  % Mean across samples
    var_features = var(features, 0, 1); % Variance across samples
    std_features = std(features, 0, 1); % Standard deviation
    
    % Parse the user and feature type from the file name
    tokens = split(files(i).name, '_');
    user_id = tokens{1}; % e.g., "U01"
    feature_type = strjoin(tokens(2:end-1), '_'); % Feature description
    
    % Group statistics by user and feature type
    if ~isfield(user_statistics, user_id)
        user_statistics.(user_id) = struct();
    end
    user_statistics.(user_id).(feature_type) = struct(...
        'mean', mean_features, ...
        'variance', var_features, ...
        'std_dev', std_features);
    
    % Group files based on the number of features
    num_features = size(features, 2);
    
    % Fix the invalid field name issue by prepending 'F'
    group_name = sprintf('F%d_features', num_features); % Prepend 'F' to make it a valid field name
    if ~isfield(grouped_files, group_name)
        grouped_files.(group_name) = {};
    end
    grouped_files.(group_name){end+1} = files(i).name;
end

% Display and compare statistics
disp('Descriptive Statistics for Each User:');
disp(user_statistics);

% Compare statistics across users
users = fieldnames(user_statistics);
mean_comparison = [];
for j = 1:length(users)
    user_data = user_statistics.(users{j});
    user_means = [];
    feature_types = fieldnames(user_data);
    for k = 1:length(feature_types)
        user_means = [user_means, mean(user_data.(feature_types{k}).mean)];
    end
    mean_comparison = [mean_comparison; user_means];
end

% Plotting the comparison of mean features
figure;
bar(mean_comparison');
title('Comparison of Mean Features Across Users');
xlabel('Feature Types');
ylabel('Mean Value');
legend(users, 'Location', 'BestOutside');
xticks(1:length(feature_types));
xticklabels(feature_types);
xtickangle(45);

% Display grouped files
disp('Grouped Files by Number of Features:');
disp(grouped_files);

% Save the processed data to a .mat file
save(fullfile(data_folder, 'processed_data.mat'), 'user_statistics', 'grouped_files');

% Define path to folder containing .mat files
data_folder = '/Users/nisayurusandaneth/Downloads/CW-Data 2'; 

% Get list of all .mat files
fileList = dir(fullfile(dataFolder, '*.mat'));

% Prepare variables for data and labels
featureMatrix = [];
labelVector = [];
genuineID = 1;  % Genuine user ID
desiredDim = 43;  % Desired feature dimension

disp('--- Loading data from the specified folder ---');

for idx = 1:length(fileList)
    matFilePath = fullfile(dataFolder, fileList(idx).name);
    currentFileName = fileList(idx).name;
    
    % Extract user ID from file name using regex
    userID = regexp(currentFileName, 'U(\d+)_', 'tokens');
    if isempty(userID)
        warning('No user ID found in file: %s', currentFileName);
        continue;
    end
    userID = str2double(userID{1}{1});
    
    disp(['Loading data for User ', num2str(userID), ' from file: ', currentFileName]);
    
    try
        % Load the data from the .mat file
        loadedData = load(matFilePath);
        dataFields = fieldnames(loadedData);
        
        if isempty(dataFields)
            warning('No data found in file: %s', currentFileName);
            continue;
        end
        
        data = loadedData.(dataFields{1});
        fprintf('Data loaded with dimensions: %s\n', mat2str(size(data)));
        
        % Ensure the data matches the target feature dimension
        if size(data, 2) > desiredDim
            data = data(:, 1:desiredDim);
        elseif size(data, 2) < desiredDim
            data = [data, zeros(size(data, 1), desiredDim - size(data, 2))];
        end
        
        % Append features and labels
        featureMatrix = [featureMatrix; data];
        
        % Labels: 1 for genuine user, 0 for impostor
        if userID == genuineID
            labelVector = [labelVector; ones(size(data, 1), 1)];
        else
            labelVector = [labelVector; zeros(size(data, 1), 1)];
        end
        
    catch err
        fprintf('Error loading file %s: %s\n', currentFileName, err.message);
    end
end

disp('--- Data loading completed ---');
disp(['Total samples: ', num2str(size(featureMatrix, 1))]);
disp(['Feature dimension: ', num2str(size(featureMatrix, 2))]);

if isempty(featureMatrix)
    error('No valid data was loaded. Please check the input files.');
end

% Basic Statistics and Visualization
disp('Performing basic statistics and visualizations...');

% Split data into genuine and impostor classes
genuineSamples = featureMatrix(labelVector == 1, :);
impostorSamples = featureMatrix(labelVector == 0, :);

% Analyze a specific feature (e.g., feature 1)
featureIndex = 1;
genuineMean = mean(genuineSamples(:, featureIndex));
genuineVar = var(genuineSamples(:, featureIndex));
impostorMean = mean(impostorSamples(:, featureIndex));
impostorVar = var(impostorSamples(:, featureIndex));

fprintf('Feature %d Analysis:\n', featureIndex);
fprintf('Genuine User: Mean = %.4f, Variance = %.4f\n', genuineMean, genuineVar);
fprintf('Impostor: Mean = %.4f, Variance = %.4f\n', impostorMean, impostorVar);

% Plot histograms for the feature
figure;
hold on;
histogram(genuineSamples(:, featureIndex), 'Normalization', 'probability');
histogram(impostorSamples(:, featureIndex), 'Normalization', 'probability');
xlabel('Feature Value');
ylabel('Probability');
legend('Genuine', 'Impostor');
title(['Feature ', num2str(featureIndex), ' Distribution']);
hold off;

% Neural Network Training
disp('Normalizing feature data...');
normalizedData = zscore(featureMatrix);

% Update labels for network (1 -> 2, 0 -> 1)
adjustedLabels = labelVector + 1;

% Split data into training and testing sets (70% train, 30% test)
disp('Creating a 70/30 train-test split...');
cv = cvpartition(adjustedLabels, 'HoldOut', 0.3);
trainData = normalizedData(training(cv), :);
trainLabels = adjustedLabels(training(cv));
testData = normalizedData(test(cv), :);
testLabels = adjustedLabels(test(cv));

disp('Initializing neural network...');
nnConfig = [64, 32];  % Define layer configuration
nnModel = patternnet(nnConfig);

disp('Training the neural network...');
[trainedNN, ~] = train(nnModel, trainData', full(ind2vec(trainLabels')));

disp('Evaluating neural network performance...');
output = trainedNN(testData');
predLabels = vec2ind(output) - 1;

accuracy = sum(predLabels' == (testLabels - 1)) / length(testLabels) * 100;
fprintf('Neural Network Accuracy: %.2f%%\n', accuracy);

% Compute precision, recall, and F1 score
tp = sum((predLabels' == 1) & ((testLabels - 1) == 1));
fp = sum((predLabels' == 1) & ((testLabels - 1) == 0));
fn = sum((predLabels' == 0) & ((testLabels - 1) == 1));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1-Score: %.2f%%\n', f1Score * 100);

figure;
confusionchart(testLabels - 1, predLabels);
title('Neural Network Confusion Matrix');

% Perform PCA for feature reduction
disp('Applying PCA for feature reduction...');

[coeffs, scores, eigenVals] = pca(trainData);
explainedVariance = cumsum(eigenVals) / sum(eigenVals);
numComponents = find(explainedVariance >= 0.95, 1);

reducedTrainData = scores(:, 1:numComponents);
reducedTestData = (testData - mean(trainData)) * coeffs(:, 1:numComponents);

disp('Retraining neural network with PCA-reduced data...');
pcaNN = patternnet(nnConfig);
[pcaTrainedNN, ~] = train(pcaNN, reducedTrainData', full(ind2vec(trainLabels')));

disp('Evaluating PCA-reduced neural network...');
pcaOutput = pcaTrainedNN(reducedTestData');
pcaPredLabels = vec2ind(pcaOutput) - 1;

pcaAccuracy = sum(pcaPredLabels' == (testLabels - 1)) / length(testLabels) * 100;
fprintf('PCA Model Accuracy: %.2f%%\n', pcaAccuracy);

% Compute metrics for the PCA model
tpPca = sum((pcaPredLabels' == 1) & ((testLabels - 1) == 1));
fpPca = sum((pcaPredLabels' == 1) & ((testLabels - 1) == 0));
fnPca = sum((pcaPredLabels' == 0) & ((testLabels - 1) == 1));

precisionPca = tpPca / (tpPca + fpPca);
recallPca = tpPca / (tpPca + fnPca);
f1ScorePca = 2 * (precisionPca * recallPca) / (precisionPca + recallPca);

fprintf('PCA Precision: %.2f%%\n', precisionPca * 100);
fprintf('PCA Recall: %.2f%%\n', recallPca * 100);
fprintf('PCA F1-Score: %.2f%%\n', f1ScorePca * 100);

figure;
confusionchart(testLabels - 1, pcaPredLabels);
title('PCA Model Confusion Matrix');

% Model comparison
fprintf('Model Comparison:\n');
fprintf('NN Accuracy: %.2f%% vs PCA NN Accuracy: %.2f%%\n', accuracy, pcaAccuracy);

% Cross-Validation for Robustness
disp('Running 5-fold cross-validation...');

cvPartition = cvpartition(length(trainLabels), 'KFold', 5);
cvAcc = zeros(cvPartition.NumTestSets, 1);
for fold = 1:cvPartition.NumTestSets
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    foldTrainData = trainData(trainIdx, :);
    foldTrainLabels = trainLabels(trainIdx);
    foldTestData = trainData(testIdx, :);
    foldTestLabels = trainLabels(testIdx);
    
    cvModel = patternnet(nnConfig);
    [cvModel, ~] = train(cvModel, foldTrainData', full(ind2vec(foldTrainLabels')));
    
    foldOutput = cvModel(foldTestData');
    foldPredLabels = vec2ind(foldOutput) - 1;
    
    cvAcc(fold) = sum(foldPredLabels' == (foldTestLabels - 1)) / length(foldTestLabels) * 100;
end

fprintf('Average Cross-Validation Accuracy: %.2f%%\n', mean(cvAcc));

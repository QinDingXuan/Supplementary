% Train the Network

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(4,64,'Padding',1,'BiasLearnRateFactor',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',3)
    convolution2dLayer(4,64,'Padding',2,'BiasLearnRateFactor',0)
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(256,'BiasLearnRateFactor',0)
    reluLayer
    fullyConnectedLayer(64,'BiasLearnRateFactor',0)
    reluLayer
    fullyConnectedLayer(10,'BiasLearnRateFactor',0)
    softmaxLayer
    classificationLayer];

digitDatasetPath_train = fullfile(pwd,'fashion_mnist_train');
imsTrain = imageDatastore(digitDatasetPath_train,'IncludeSubfolders',true,'LabelSource','foldernames');

digitDatasetPath_test = fullfile(pwd,'fashion_mnist_test');
imsValidation = imageDatastore(digitDatasetPath_test,'IncludeSubfolders',true,'LabelSource','foldernames');

YValidation = imsValidation.Labels;

options = trainingOptions('sgdm','InitialLearnRate',0.00025,'MaxEpochs',400, ...
    'Shuffle','every-epoch','ValidationFrequency',2000,'ExecutionEnvironment','auto', ...
    'Verbose',false,'Plots','training-progress','MiniBatchSize',128,'ValidationData',...
    imsValidation,'LearnRateDropPeriod',180,'LearnRateDropFactor',0.25,'LearnRateSchedule','piecewise');

net = trainNetwork(imsTrain,layers,options);

% Testing the robustness of neural network weights
% i.e. the prediction accuracy after adding noise

load('net_pretrain.mat') % Load the trained net

N = 300;

testPred = classify(net,imsValidation);

CNNAccuracy = sum(testPred == YValidation)/numel(YValidation)*100;

fprintf('Baseline Accuracy =  %f:\n\n',LSTMAccuracy);

Max_Bias = 1.0;

noise = [0.1 0.2 0.3 0.5 0.7];

for i = 1:5

    accuracy_noise_modifyNtimes = zeros(1,N);
 
    delta = noise(i);

    for j = 1:N
        modify_net = net.saveobj; 

        RAND = randn(size(net.Layers(2,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;
        modify_net.Layers(2,1).Weights = net.Layers(2,1).Weights+delta.*RAND.*net.Layers(2,1).Weights;
      
        RAND = randn(size(net.Layers(5,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;
        modify_net.Layers(5,1).Weights = net.Layers(5,1).Weights+delta.*RAND.*net.Layers(5,1).Weights;

        RAND = randn(size(net.Layers(8,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;
        modify_net.Layers(8,1).Weights = net.Layers(8,1).Weights+delta.*RAND.*net.Layers(8,1).Weights;
        
        RAND = randn(size(net.Layers(10,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;
        modify_net.Layers(10,1).Weights = net.Layers(10,1).Weights+delta.*RAND.*net.Layers(10,1).Weights;
       
        RAND = randn(size(net.Layers(12,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;
        modify_net.Layers(12,1).Weights = net.Layers(12,1).Weights+delta.*RAND.*net.Layers(12,1).Weights;

        modify_net = net.loadobj(modify_net);

        YPred_noiseNtimes = classify(modify_net,imsValidation);
        accuracy_noise_modifyNtimes(j) = sum(YPred_noiseNtimes == YValidation)/numel(YValidation);
    end
    fprintf('delta =  %f:\n',delta);
    accuracy_noiseNtimes_final = sum(accuracy_noise_modifyNtimes) / N;
    fprintf('Mean Value: %f\n',accuracy_noiseNtimes_final);
    fprintf('Min Value: %f\n',min(accuracy_noise_modifyNtimes));
    fprintf('Max Value: %f\n',max(accuracy_noise_modifyNtimes));
    fprintf('Variance: %f\n\n',sqrt(var(100*accuracy_noise_modifyNtimes)));
end
% Train the Network

load('ECG_Data.mat')

layers = [ ...
    sequenceInputLayer(2)
    bilstmLayer(200,'OutputMode','last')
    batchNormalizationLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('sgdm','InitialLearnRate',0.1,'MaxEpochs',250, ...
    'Shuffle','every-epoch','ValidationFrequency',345,'ExecutionEnvironment','auto', ...
    'Verbose',false,'Plots','training-progress','MiniBatchSize',128,'ValidationData',...
    {XTest , YTest },'LearnRateDropPeriod',80,'LearnRateDropFactor',0.1,...
    'LearnRateSchedule','piecewise','Momentum',0.9,'ResetInputNormalization',true,...
    'L2Regularization',1e-4,'OutputNetwork','best-validation-loss');

net = trainNetwork(XTrain,YTrain,layers,options);


% Testing the robustness of neural network weights
% i.e. the prediction accuracy after adding noise
load('net_pretrain.mat') % Load the trained net

testPred = classify(net,XTest);

LSTMAccuracy = sum(testPred == YTest)/numel(YTest)*100;

fprintf('Baseline Accuracy =  %f:\n\n',LSTMAccuracy);

N = 300;

Max_Bias = 1.0;

for delta = 0.2:0.2:1.0

    accuracy_noise_modifyNtimes = zeros(1,N);

    for i = 1:N
        modify_net = net.saveobj; 

        RAND = randn(size(net.Layers(2,1).InputWeights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;        
        modify_net.Layers(2,1).InputWeights = net.Layers(2,1).InputWeights+...
            delta.*RAND.*net.Layers(2,1).InputWeights;     

        RAND = randn(size(net.Layers(2,1).RecurrentWeights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;        
        modify_net.Layers(2,1).RecurrentWeights = net.Layers(2,1).RecurrentWeights+...
            delta.*RAND.*net.Layers(2,1).RecurrentWeights;    

        RAND = randn(size(net.Layers(4,1).Weights));
        RAND(RAND>Max_Bias) = Max_Bias;
        RAND(RAND<-Max_Bias) = -Max_Bias;        
        modify_net.Layers(4,1).Weights = net.Layers(4,1).Weights+...
            delta.*RAND.*net.Layers(4,1).Weights;        

        modify_net = net.loadobj(modify_net);

        YPred_noiseNtimes = classify(modify_net,XTest);
        accuracy_noise_modifyNtimes(i) = sum(YPred_noiseNtimes == YTest)/numel(YTest);
    end
    fprintf('delta =  %f:\n',delta);
    accuracy_noiseNtimes_final = sum(accuracy_noise_modifyNtimes) / N;
    fprintf('Mean Value: %f\n',accuracy_noiseNtimes_final);
    fprintf('Min Value: %f\n',min(accuracy_noise_modifyNtimes));
    fprintf('Max Value: %f\n',max(accuracy_noise_modifyNtimes));
    fprintf('Variance: %f\n\n',sqrt(var(100*accuracy_noise_modifyNtimes)));
end
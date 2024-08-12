% Train the Network

layers = [
    imageInputLayer([32 32 3])

    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer('Name','relu_1_1')



    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_1_1')
    reluLayer('Name','relu_1_2')

    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_1_2')
    reluLayer('Name','relu_1_3')

    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,16,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_1_3')
    reluLayer('Name','relu_2_1')



    convolution2dLayer(3,32,'Padding',1,'Stride',2,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_2_1')
    reluLayer('Name','relu_2_2')

    convolution2dLayer(3,32,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_2_2')
    reluLayer('Name','relu_2_3')

    convolution2dLayer(3,32,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_2_3')
    reluLayer('Name','relu_3_1')



    convolution2dLayer(3,64,'Padding',1,'Stride',2,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_3_1')
    reluLayer('Name','relu_3_2')

    convolution2dLayer(3,64,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_3_2')
    reluLayer('Name','relu_3_3')

    convolution2dLayer(3,64,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Padding',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    additionLayer(2,'Name','add_3_3')
    dropoutLayer(0.2)
    reluLayer('Name','relu_4_1')


    convolution2dLayer(2,64,'Padding',0,'Stride',2,'BiasLearnRateFactor',0)
    batchNormalizationLayer
    reluLayer('Name','avgpool2d')

    convolution2dLayer(3,32,'Padding',1,'Stride',2,'BiasLearnRateFactor',0,'Name','conv1_2')
    batchNormalizationLayer('Name','bn1_2')

    convolution2dLayer(3,64,'Padding',1,'Stride',2,'BiasLearnRateFactor',0,'Name','conv2_3')
    batchNormalizationLayer('Name','bn2_3')

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

lgraph = layerGraph(layers);

lgraph = disconnectLayers(lgraph,'avgpool2d','conv1_2');
lgraph = disconnectLayers(lgraph,'bn1_2','conv2_3');
lgraph = disconnectLayers(lgraph,'bn2_3','fc');
lgraph = connectLayers(lgraph,'avgpool2d','fc');
lgraph = connectLayers(lgraph,'relu_1_1','add_1_1/in2');
lgraph = connectLayers(lgraph,'relu_1_2','add_1_2/in2');
lgraph = connectLayers(lgraph,'relu_1_3','add_1_3/in2');
lgraph = connectLayers(lgraph,'relu_2_1','conv1_2');
lgraph = connectLayers(lgraph,'bn1_2','add_2_1/in2');
lgraph = connectLayers(lgraph,'relu_2_2','add_2_2/in2');
lgraph = connectLayers(lgraph,'relu_2_3','add_2_3/in2');
lgraph = connectLayers(lgraph,'relu_3_1','conv2_3');
lgraph = connectLayers(lgraph,'bn2_3','add_3_1/in2');
lgraph = connectLayers(lgraph,'relu_3_2','add_3_2/in2');
lgraph = connectLayers(lgraph,'relu_3_3','add_3_3/in2');


imsTrain = single(zeros(32,32,3,50000));

imsTest = single(zeros(32,32,3,10000));

load('data_batch_1.mat') 

imsTrain_Label = labels;

for i = 1:10000


    imsTrain(:,:,:,i) = reshape(data(i,:),[32,32,3]);

    imsTrain(:,:,1,i) = imsTrain(:,:,1,i)';

    imsTrain(:,:,2,i) = imsTrain(:,:,2,i)';

    imsTrain(:,:,3,i) = imsTrain(:,:,3,i)';

end

load('data_batch_2.mat') 

imsTrain_Label = [imsTrain_Label ; labels];

for i = 10001:20000

    imsTrain(:,:,:,i) = reshape(data(i-10000,:),[32,32,3]);

    imsTrain(:,:,1,i) = imsTrain(:,:,1,i)';

    imsTrain(:,:,2,i) = imsTrain(:,:,2,i)';

    imsTrain(:,:,3,i) = imsTrain(:,:,3,i)';

end

load('data_batch_3.mat') 

imsTrain_Label = [imsTrain_Label ; labels];

for i = 20001:30000

    imsTrain(:,:,:,i) = reshape(data(i-20000,:),[32,32,3]);

    imsTrain(:,:,1,i) = imsTrain(:,:,1,i)';

    imsTrain(:,:,2,i) = imsTrain(:,:,2,i)';

    imsTrain(:,:,3,i) = imsTrain(:,:,3,i)';

end

load('data_batch_4.mat') 

imsTrain_Label = [imsTrain_Label ; labels];
for i = 30001:40000

    imsTrain(:,:,:,i) = reshape(data(i-30000,:),[32,32,3]);

    imsTrain(:,:,1,i) = imsTrain(:,:,1,i)';

    imsTrain(:,:,2,i) = imsTrain(:,:,2,i)';

    imsTrain(:,:,3,i) = imsTrain(:,:,3,i)';

end

load('data_batch_5.mat') 

imsTrain_Label = [imsTrain_Label ; labels];

for i = 40001:50000

    imsTrain(:,:,:,i) = reshape(data(i-40000,:),[32,32,3]);

    imsTrain(:,:,1,i) = imsTrain(:,:,1,i)';

    imsTrain(:,:,2,i) = imsTrain(:,:,2,i)';

    imsTrain(:,:,3,i) = imsTrain(:,:,3,i)';

end

load('test_batch.mat') 

imsTest_Label = labels;

imsTrain_Label = categorical(imsTrain_Label);

imsTest_Label = categorical(imsTest_Label);

for i = 1:10000

    imsTest(:,:,:,i) = reshape(data(i,:),[32,32,3]);

    imsTest(:,:,1,i) = imsTest(:,:,1,i)';

    imsTest(:,:,2,i) = imsTest(:,:,2,i)';

    imsTest(:,:,3,i) = imsTest(:,:,3,i)';

end

options = trainingOptions('sgdm','InitialLearnRate',0.1,'MaxEpochs',250, ...
    'Shuffle','every-epoch','ValidationFrequency',1950,'ExecutionEnvironment','auto', ...
    'Verbose',false,'Plots','training-progress','MiniBatchSize',128,'ValidationData',...
    {imsTest , imsTest_Label },'LearnRateDropPeriod',80,'LearnRateDropFactor',0.1,...
    'LearnRateSchedule','piecewise','Momentum',0.9,'L2Regularization',1e-4,...
    'OutputNetwork','last-iteration');

net = trainNetwork(imsTrain,imsTrain_Label,lgraph,options);


 
% Testing the robustness of neural network weights
% i.e. the prediction accuracy after adding noise

load('net_pretrain.mat') % Load the trained net

N = 300;

testPred = classify(net,imsTest);

CNNAccuracy = sum(testPred == imsTest_Label)/numel(imsTest_Label);

fprintf('Baseline Accuracy =  %f:\n\n',CNNAccuracy);

Max_Bias = 1.0;

for delta = 0.2:0.2:1.0

    accuracy_noise_modify100times = zeros(1,N);

    Layer_Num = [2 5 8 12 15 19 22 26 29 33 36 40 43 47 50 54 57 61 64 69 72 74 76];

    for i = 1:N
        modify_net = net.saveobj; 

        for num = 1:length(Layer_Num)
            RAND = randn(size(net.Layers(Layer_Num(num),1).Weights));
            RAND(RAND>Max_Bias) = Max_Bias;
            RAND(RAND<-Max_Bias) = -Max_Bias;        
            modify_net.Layers(Layer_Num(num),1).Weights = net.Layers(Layer_Num(num),1).Weights+delta.*RAND.*net.Layers(Layer_Num(num),1).Weights;
        end                 

        modify_net = net.loadobj(modify_net);

        YPred_noiseNtimes = classify(modify_net,imsTest);
        accuracy_noise_modify100times(i) = sum(YPred_noiseNtimes == imsTest_Label)/numel(imsTest_Label);
    end
    fprintf('delta =  %f:\n',delta);
    accuracy_noise100times_final = sum(accuracy_noise_modify100times) / N;
    fprintf('Mean Accuracy: %f\n',accuracy_noise100times_final);
    fprintf('Min Accuracy: %f\n',min(accuracy_noise_modify100times));
    fprintf('Max Accuracy: %f\n',max(accuracy_noise_modify100times));
    fprintf('Variance: %f\n\n',sqrt(var(100*accuracy_noise_modify100times)));
end
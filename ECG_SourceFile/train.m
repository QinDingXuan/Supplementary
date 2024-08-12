function net = train(this, net, data)
    % train   Train a network
    %
    % Inputs
    %    net -- network to train
    %    data -- data encapsulated in a data dispatcher
    % Outputs
    %    net -- trained network
    reporter = this.Reporter;
    schedule = this.Schedule;
    prms = collectSettings(this, net);
    outputStrategy = this.OutputStrategy;
    data.start();
    this.Summary = this.SummaryFcn(net.OutputFormats, prms.maxEpochs);

    regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);

    solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);

    trainingTimer = tic;

    % Set StopTrainingFlag and StopReason before starting reporters
    % since they might stop training
    this.StopTrainingFlag = false;
    this.StopReason = nnet.internal.cnn.util.TrainingStopReason.FinalIteration;
    reporter.start();
    iteration = 0;
    gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
        'Threshold', this.Options.GradientThreshold);
    gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
    learnRate = initializeLearning(this);           

    delta = gpuArray(single(0.8));

    add_times = 100;

    for epoch = 1:prms.maxEpochs
        this.shuffle( data, prms.shuffleOption, epoch );
        data.start();
        while ~data.IsDone && ~this.StopTrainingFlag
            [X, T, info] = data.next();
            % Cast data to appropriate execution environment for
            % training and apply transforms
            X = this.ExecutionStrategy.environment(X);
            T = this.ExecutionStrategy.environment(T);

            initial_net = net;

            for times = 1:add_times 

                if mod(times,2) == 1
                    for num = 1:2
                        RAND = delta * randn(size(initial_net.Layers{2,1}.LearnableParameters(1,num).Value));
                        Noise1{num} = gpuArray(single(RAND)).*initial_net.Layers{2,1}.LearnableParameters(1,num).Value;
                        net.Layers{2,1}.LearnableParameters(1,num).Value = initial_net.Layers{2,1}.LearnableParameters(1,num).Value + Noise1{num}; 
                        Jacobian{num} = 1 +  gpuArray(single(RAND));
                    end
                    RAND = delta * randn(size(initial_net.Layers{4,1}.LearnableParameters(1,1).Value));
                    Noise2 = gpuArray(single(RAND)).*initial_net.Layers{4,1}.LearnableParameters(1,1).Value;
                    net.Layers{4,1}.LearnableParameters(1,1).Value = initial_net.Layers{4,1}.LearnableParameters(1,1).Value + Noise2;            
                    Jacobian{3} = 1 +  gpuArray(single(RAND));
                else
                    for num = 1:2
                        net.Layers{2,1}.LearnableParameters(1,num).Value = initial_net.Layers{2,1}.LearnableParameters(1,num).Value - Noise1{num}; 
                        Jacobian{num} = 2 -  Jacobian{num};
                    end
                        net.Layers{4,1}.LearnableParameters(1,1).Value = initial_net.Layers{4,1}.LearnableParameters(1,1).Value - Noise2;
                        Jacobian{3} = 2 -  Jacobian{3};
                end

                [gradients, Y, states] = this.computeGradients(net, X, T);

                gradients = thresholdGradients(gradThresholder,gradients);

                if times == 1
                    miniBatchLoss = net.loss(Y,T);
                    gradients{1} = gradients{1}.*Jacobian{1};
                    gradients{2} = gradients{2}.*Jacobian{2};
                    gradients{6} = gradients{6}.*Jacobian{3};
                    gradients_sum = gradients;
                elseif times > 1
                    miniBatchLoss = miniBatchLoss + net.loss(Y,T);
                    gradients{1} = gradients{1}.*Jacobian{1};
                    gradients{2} = gradients{2}.*Jacobian{2};
                    gradients{6} = gradients{6}.*Jacobian{3};                            
                    for column = 1:size(gradients,2)
                        gradients_sum{column} = gradients_sum{column} + gradients{column};
                    end
                end

            end 

            for column = 1:size(gradients,2)

                gradients{column} =  gradients_sum{column} / add_times;

            end   

            for num = 1:2

                net.Layers{2,1}.LearnableParameters(1,num).Value = initial_net.Layers{2,1}.LearnableParameters(1,num).Value;
                net.Layers{4,1}.LearnableParameters(1,num).Value = initial_net.Layers{4,1}.LearnableParameters(1,num).Value;

            end

            % Compute the average loss

            miniBatchLoss = miniBatchLoss / add_times;

            gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);

            gradients = thresholdGradients(gradThresholder,gradients);

            velocity = solver.calculateUpdate(gradients,learnRate);

            net = net.updateLearnableParameters(velocity);

            net = this.setNetworkState(net, states, info.IsCompleteObservation);

            elapsedTime = toc(trainingTimer);

            iteration = iteration + 1;
            this.Summary.update(Y, T, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, data.IsDone);
            % It is important that computeIteration is called
            % before reportIteration, so that the summary is
            % correctly updated before being reported
            reporter.computeIteration( this.Summary, net );
            reporter.reportIteration( this.Summary, net );

            outputStrategy.updateOutputNetwork(this.Summary, net);
        end
        learnRate = schedule.update(learnRate, epoch);

        reporter.reportEpoch( epoch, iteration, net );

        % If an interrupt request has been made, break out of the
        % epoch loop
        if this.StopTrainingFlag
            % If the summary has not been updated, set its
            % Iteration count to zero before breaking out of the
            % epoch loop. This ensures subsequent Reporter methods
            % can be called.
            if isempty(this.Summary.Iteration)
                this.Summary.Iteration = 0;
            end

            break;
        end
    end
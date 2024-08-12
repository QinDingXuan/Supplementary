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
    add_times = 10;

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

            Layer_Num = [2 5 8 12 15 19 22 26 29 31 35 38 42 45 49 52 54 58 61 65 68 73 76];
            layer2 = 1:4:89;

            for times = 1:add_times 
                if mod(times,2) == 1
                    for num = 1:length(Layer_Num)
                        RAND = delta * randn(size(initial_net.Layers{Layer_Num(num),1}.Weights.Value));
                        Jacobian{num} = 1 +  gpuArray(single(RAND));
                        Noise{num} = gpuArray(single(RAND)).*initial_net.Layers{Layer_Num(num),1}.Weights.Value;
                        net.Layers{Layer_Num(num),1}.Weights.Value = initial_net.Layers{Layer_Num(num),1}.Weights.Value + Noise{num};
                    end
                else
                    for num = 1:length(Layer_Num)
                        net.Layers{Layer_Num(num),1}.Weights.Value = initial_net.Layers{Layer_Num(num),1}.Weights.Value - Noise{num};
                        Jacobian{num} = 2 - Jacobian{num};
                    end
                end
              
                [gradients, Y, states] = this.computeGradients(net, X, T);

                if times == 1
                    miniBatchLoss = net.loss(Y,T);
                    for column = 1:size(layer2)                              
                        gradients{layer2(column)} = gradients{layer2(column)}.*Jacobian{column};                             
                    end   
                    gradients_sum = gradients;
                elseif times > 1
                    miniBatchLoss = miniBatchLoss + net.loss(Y,T);
                    for column = 1:size(layer2)
                        gradients{layer2(column)} = gradients{layer2(column)}.*Jacobian{column};                               
                    end                                     
                    for column = 1:size(gradients,2)                               
                        gradients_sum{column} = gradients_sum{column} + gradients{column};
                    end
                end
            end 

            for column = 1:size(gradients,2)
              gradients{column} =  gradients_sum{column} / (add_times);
            end   
       
            for num = 1:length(Layer_Num)
                net.Layers{Layer_Num(num),1}.Weights.Value = initial_net.Layers{Layer_Num(num),1}.Weights.Value;
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
    reporter.computeFinish( this.Summary, net );
    reporter.finish( this.Summary, this.StopReason );
    % Validation is always computed at the end of training if it
    % hasn't been computed in the last iteration. Update output
    % network before getting output network
    outputStrategy.updateOutputNetwork(this.Summary, net);
    net = outputStrategy.getOutputNetwork( net );
    % Add the iteration corresponding to net to summary so
    % that any reporter that needs the output network iteration at
    % reportFinalIteration will have access to it
    outputStrategy.addNetworkIterationToSummary( this.Summary )
end
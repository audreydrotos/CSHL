%% Reduce the values of trial for PCA
stepSize = 10;
idx = 1:stepSize:size(smoothedTensor,2);

tensorPCA = smoothedTensor(:,idx,:);
%allScores = [];
figure
for rr = 1:length(regionSelected)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);
    
    % Generate PCA
    nexttile(rr+length(regionSelected))
   
    color_sequence = parula(valueNum); 
    % find trials we want to summarize
    % behavior_idx = trial.
    % averageTrials = mw
    %
    averageTrials = mean(region_neurons,3);
    % averageBins = mean(region_neurons,2);
    % averageBins = squeeze(averageBins);
    % Run the PCA
    [coefs, scores, ~, ~, explained ] = pca(averageTrials');
    %[coefs, scores, ~, ~, explained ] = pca(averageBins');
    cumulative_variance = cumsum(explained)/100;
    K = find(cumulative_variance >= 0.80, 1); % choose K

    %plot Scree-plot to decide K
    plot(1:length(explained), explained, 'bo-', 'LineWidth', 2);
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    title(['Scree Plot for PCA on',regions.name(regionSelected(rr))],[num2str(K),' component count for 80% variance']);
    xlim([0 10])
    grid on;
    %allScores(:,:,rr) = scores;
    % separate pca; can't draw together
    nexttile(rr+2*length(regionSelected))
    plot3(scores(:,1), scores(:,2), scores(:,3),colorSelected(rr));
    xlabel('PCA 1')
    ylabel('PCA 2')
    zlabel('PCA 3')
    title('PCA on',regions.name(region_code))
 
end

figure
for rr = 1:length(regionSelected)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);
    
    % Generate PCA
    nexttile(rr+length(regionSelected))
    
    % find trials we want to summarize
    
    %averageTrials = mean(region_neurons,3);
    averageBins = mean(region_neurons,2);
    averageBins = squeeze(averageBins);
    % Run the PCA
    %[coefs, scores, ~, ~, explained ] = pca(averageTrials');
    [coefs, scores, ~, ~, explained ] = pca(averageBins');
    cumulative_variance = cumsum(explained)/100;
    K = find(cumulative_variance >= 0.80, 1); % choose K

    %plot Scree-plot to decide K
    plot(1:length(explained), explained, 'bo-', 'LineWidth', 2);
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    title(['Scree Plot for PCA on',regions.name(regionSelected(rr))],[num2str(K),' component count for 80% variance']);
    xlim([0 10])
    grid on;
    %allScores(:,:,rr) = scores;
    % separate pca; can't draw together
    nexttile(rr+2*length(regionSelected))
    %plot different color to different trials
    color_sequence = cool(valueNum);  
    for i = 1:valueNum
        currentScore = scores(behavior==behavior_value(i),:);
        scatter3(currentScore(:,1), currentScore(:,2),currentScore(:,3),50,color_sequence(i,:),'filled');
        hold on
    end
    
    xlabel('PCA 1')
    ylabel('PCA 2')
    zlabel('PCA 3')
    title('PCA on',regions.name(region_code))
 
end
% nexttile([1, 2]);
% for rr = 1:2
%     % plot coef by signals
%     plot3(allScores(:,1,rr), allScores(:,2,rr), allScores(:,3,rr),colorSelected(rr));
%     hold on
% end
% legend([regions.name(regionSelected)]);
saveas(gcf, ['figure/' sesPath '_pca.fig']);

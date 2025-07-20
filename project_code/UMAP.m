%% Umap
% Use 'run_umap' to reduce the dim to 3
% try different values for n_neighbors ranging from 5 to 199
% Umap need to classify neurons
n_components = 2;
n_neighbors = 20;
figure;
t = tiledlayout(1, length(regionSelected)); 

for rr = 1:length(regionSelected)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    
     % Generate UMAP
    nexttile(rr)
    region_neurons = binnedTensor(region_idx, :, :);

    summedTensor = sum(region_neurons,2);
    neuronNum = size(region_neurons,1);
    
    firing_Onbehavior = zeros(valueNum,neuronNum);
    time_Onbehavior = zeros(valueNum,neuronNum);

    for i = 1:neuronNum
        for j = 1:nTrials
            for v = 1:valueNum
                if behavior(j)==behavior_value(v)
                    firing_Onbehavior(v,i) = firing_Onbehavior(v,i)+summedTensor(i, 1, j);
                    time_Onbehavior(v,i) = time_Onbehavior(v,i)+1;
                end
            end
        end
    end

    firingRate_Onbehavior = firing_Onbehavior./time_Onbehavior;
    [peak_values, peak_stimuli] = max(firingRate_Onbehavior, [], 1);

    % find
    region_neurons = tensorPCA(region_idx, :, :);
    averageTrials = mean(region_neurons,3);
    rng(42);
    [rep_UMAP, umap, clusterIdentifiers, extras]=run_umap(double(averageTrials), ...
    'n_components', n_components, 'n_neighbors', n_neighbors, 'verbose', 'none');
    % plot coef by signals
    % plot using different neurons
    color_sequence = parula(valueNum); 

    for i = 1:valueNum
        currentUmap= rep_UMAP(peak_stimuli==i,:);
        scatter(currentUmap(:,1), currentUmap(:,2),'filled');
        hold on
    end
    
    xlabel('UMAP 1')
    ylabel('UMAP 2')
    %zlabel('UMAP 3')
    title('UMAP on',regions.name(region_code))

end

saveas(gcf, ['figure/' sesPath '_umap.fig']);
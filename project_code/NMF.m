%% NMF
nFactors = 5;
figure;
t = tiledlayout(1, length(regionSelected)); 

for rr = 1:length(regionSelected)
    region_code = regionSelected(rr);
    region_idx = neurons.region == region_code;
    region_neurons = tensorPCA(region_idx, :, :);
     % Generate NMF
    nexttile(rr)

    % find
    averageTrials = mean(region_neurons,3);
   
    % this is the nmf
    [W,H, disc] = nnmf(averageTrials, nFactors); % specify six factors
    myNMFLoads = H';
    % plot coef by signals
    plot3(myNMFLoads(:,1), myNMFLoads(:,2), myNMFLoads(:,3),colorSelected(rr));
    xlabel('C 1'); ylabel('C 2'); zlabel('C 3');
    title('NMF on',regions.name(region_code))
    hold on

end

saveas(gcf, ['figure/' sesPath '_nmf.fig']);
function fig6 = createFigure6(data, pathResults)
% Reproduces Figure 6 of the original paper (reconstruction over
% undersampled and fully sampled cardiac data). It first sets properties
% that are held constant for all of the following reconstructions. They are
% very similar to the main script. Then reconstructions using only
% 11, 22, 33, 44 equidistant or all 55 spokes of the provided heart dataset are
% done (equals undersampling factors of 5,4,3,2,1 respectively).
                
% general properties for all recons
properties.image_dim = data.Nimg;
properties.gridding.oversampling_factor = data.overgrid_factor;
properties.gridding.kernel_width = 5;
properties.visualization_level = 1;
properties.do_sense_recon = 1;
properties.n_iterations = 10;

% setup property array for recon loop
nSpokesArray = 11*(1:5);
nRecons = numel(nSpokesArray);

% output Array
cardiacImages = zeros(properties.image_dim, properties.image_dim, nRecons);

%% Loop performing Recons with different undersampling factors
for iRecon = 1:nRecons
    fprintf('Reconstruct Image with %d spokes... (%d/%d)\n', ...
        nSpokesArray(iRecon), iRecon, nRecons);
    % give index array of selected spokes explicitly
    properties.undersampling_factor = 1:nSpokesArray(iRecon);
    out = CGSense(data, properties);
    cardiacImages(:,:,iRecon) = out.imageComb;    
end
save('result_heart.mat', 'cardiacImages');

%% Plot the results and save the figure in results subfolder
fig6 = figure('Name', 'Figure 6: Heart Recon with Various Radial Undersampling Factors');
hs = zeros(1,nRecons);
for iRecon = 1:nRecons
    hs(iRecon) = subplot(1,5,iRecon); imagesc(abs(cardiacImages(:,:,iRecon)));
    title([num2str(nSpokesArray(iRecon)) ' spokes']); colormap(gray); axis image; axis off;
end
linkaxes(hs, 'xy'); % link axes to enable zooming for comparison

saveas(fig6,[pathResults '/Figure6_undersamplingRecon.png'])

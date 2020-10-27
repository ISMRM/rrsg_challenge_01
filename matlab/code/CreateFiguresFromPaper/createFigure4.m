function [Deltas, deltas, durationIterSteps, fig4]= createFigure4(data, pathResults)
% Reproduces Figure 4 of the original paper (log(delta) over number of iterations)
% It first sets properties that are held constant for
% all of the following reconstructions. They are very similar to the main
% script. In the first step a reconstruction using all data (R=1) is made
% with 5 iterations. The number of iterations was chosen manually as we
% found that the brain dataset showed a good reconstruction after 5 steps.
% Then, reconstructions with undersampling factors of R=2,3,4,5 are done. 
% Both relative data error "delta" and error to reference image "Delta" (R=1
% reconstruction) are stored for each iteration step.

%% General Properties
properties.image_dim = data.Nimg;        
properties.gridding.oversampling_factor = data.overgrid_factor;  
properties.gridding.kernel_width = 5;   
properties.visualization_level = 0;

%% Reconstruct R=1 image (reference for "Delta")
properties.do_sense_recon = 1;
properties.undersampling_factor = 1;
properties.n_iterations = 10;
properties.kSpaceFilterMethod = 'gridding'; 

out = CGSense(data, properties);
reference.image = out.imageComb;
mask_tmp = zeros(size(reference.image));
mask_tmp(abs(reference.image) > mean(mean(abs(reference.image)))) = 1;
se = strel('diamond',2);
mask_tmp = imopen(mask_tmp, se);
reference.mask = mask_tmp;

%% Reconstruct Range of undersampling factors (R = 2...5)
Rrange = [1, 2, 3, 4, 5];
nIterations = 10;
properties.n_iterations = nIterations;

deltas = zeros(nIterations+1, length(Rrange));
Deltas = zeros(nIterations+1, length(Rrange));
durationIterSteps = zeros(nIterations+1, length(Rrange));
for R=1:length(Rrange)
    disp(['Reconstruct image with R = ' num2str(Rrange(R)) ' (' num2str(R) '/' num2str(length(Rrange)) ')']);
    properties.undersampling_factor = Rrange(R);
    out_tmp = CGSense(data, properties, reference);
    deltas(:,R) = out_tmp.deltas;
    Deltas(:,R) = out_tmp.Deltas;
    durationIterSteps(:,R) = out_tmp.durationIterSteps;
end

%% Plot the results and save the figure in results subfolder
% Note: We decided to not plot the error for R=1
fig4 = figure('Name', 'Relative reconstruction Errors delta (to data) and Delta (to reference image) for different undersampling factors');

% small delta
subplot(1,2,1);
line1 = plot(0:nIterations,log10(Deltas(:,2:5)), 'LineWidth', 2);
xlim([0 nIterations])
ylabel('$\log_{10} \Delta_{approx}$', 'Interpreter', 'latex', 'FontSize', 16)
xlabel('Iterations', 'Interpreter', 'latex', 'FontSize', 16)
legend({'R=2', 'R=3', 'R=4', 'R=5'}, 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 14)
box on

% capital Delta
subplot(1,2,2);
line2 = plot(0:nIterations,log10(deltas(:,1:5)), 'LineWidth', 2);
xlim([0 nIterations])
ylabel('$\log_{10} \delta$', 'Interpreter', 'latex', 'FontSize', 16)
xlabel('Iterations', 'Interpreter', 'latex', 'FontSize', 16)
legend({'R=1', 'R=2', 'R=3', 'R=4', 'R=5'}, 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 14)
box on

% Make sure that same R-factors have same line color in both plots
colors = get(line2, {'Color'});
set(line1,{'Color'}, colors(2:5));

saveas(fig4,[pathResults '/Figure4_logDeltaOverIterations.png'])

end
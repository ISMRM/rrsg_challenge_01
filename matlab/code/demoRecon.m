% This script runs a sample reconstruction using the CG-SENSE algorithm.

%% Load Data
% Put in folder containing the h5 files, i.e.
pathData = '../../../CGSENSE_challenge_sub/data/cardiac_radial_KI.h5';
data = loadData(pathData);
% data.Nimg = 240;
%% Set up properties
properties.image_dim                    = data.Nimg;    % Number of voxels (assumes quadratic images)
properties.gridding.oversampling_factor = data.overgrid_factor;            % Gridding oversampling factor
properties.gridding.kernel_width        = 5;            % Gridding kernel width as a multiple of dk without oversampling
properties.do_sense_recon               = 1;            % 1 = Perform recon with SENSE maps; 0 = No sense maps used; basically iterative gridding+FFT (density-compensation)
properties.undersampling_factor         = 1;            % undersampling or acceleration factor (R), determines how fraction (1/R) of full data is used in reconstruction
properties.n_iterations                 = 10;            % Number of CG iterations
% Visualization level:
% 0 = none; 
% 1 = Plot current image after each CG iteration? 
% 2 = Plot diagnostics/aux data (sense map, k-space filter, intensity
%     correction)
properties.visualization_level          = 1;

%% Reconstruct Image
out = CGSense(data, properties);

%% Additional output figure with intermediate vs final iteration results
nIt = properties.n_iterations;
nItHalf = round(nIt/2);
fh = figure('Name', 'demoRecon: Iteration Results');
subplot(1,3,1); imagesc(abs(out.imagesIterSteps{1})); colormap(gray); axis image; axis off; ylabel('R=1'); title('iteration 0');
subplot(1,3,2); imagesc(abs(out.imagesIterSteps{nItHalf})); colormap(gray); axis image; axis off; title(sprintf('iteration %d', nItHalf));
subplot(1,3,3); imagesc(abs(out.imagesIterSteps{nIt})); colormap(gray); axis image; axis off; title(sprintf('iteration %d', nIt));

%% Single Coil
dataTmp = data;
properties.n_iterations = 1;
properties.do_sense_recon = 0;
dataTmp.sense.data = data.sense.data(:,:,1);
dataTmp.sense.noiseCovarianceMatrix = 1; % ignore noise covariance
dataTmp.signal = data.signal(:,:,1);
dataTmp.nCoils = 1;
outSingle = CGSense(dataTmp, properties);


%% Save results
initial = out.imagesIterSteps{1};
final = out.imageComb;
singleCoil = outSingle.imageComb;
save('result_cardiac.mat' , 'initial', 'final', 'singleCoil');
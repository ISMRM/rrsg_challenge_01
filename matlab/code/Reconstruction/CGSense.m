function out = CGSense(data, properties, referenceImage)
% iterativeRecon Perform CG-SENSE image reconstruction
%
% USE
%   out = CGSense(data, properties, [referenceImage])
%
% IN
%   data      	 Structure that contains the trajectory and signal
%     .signal      Complex MR signal values
%                    (nSamplesPerRFPulse x nRFPulses x nCoils)
%     .k_scaled    K-space trajectory, scaled to interval [-0.5,0.5]
%                    (nSamplesPerRFPulse x nRFPulses x 2)
%     .nCoils      Number of coils
%     .sense.data  Sensitivity profiles of the coils
%                    (nImgOs x nImgOs x nCoils)
%     .sense.mask  Mask for reconstructed image
%                    (nImgOs x nImgOs)
%
%   properties   Structure that contains the settings for CG-SENSE
%                As created in main.m
%                with the following fields
%     .image_dim                    % Number of image voxels (assumes quadratic images)
%     .gridding.oversampling_factor % Gridding oversampling factor
%     .gridding.kernel_width        % Gridding kernel width as a multiple of dk without oversampling
%     .do_sense_recon               % 1 = Perform recon with SENSE maps; 
%                                   % 0 = No sense maps used; basically iterative gridding+FFT (density-compensation)
%     .undersampling_factor         % undersampling or acceleration factor (R),
%                                   % determines which fraction (1/R) of 
%                                   % complete data is used in reconstruction
%                                   % NOTE: If more than one value is given,
%                                   % this is assumed to be the index array 
%                                   % for selected interleaves
%     .n_iterations                 % Number of CG iterations
%     .visualization_level          % Visualization level: 
%                                   % 0 = none; 
%                                   % 1 = Plot current image after each CG iteration? 
%                                   % 2 = Plot diagnostics/aux data (sense map, k-space filter, intensity
%                                                       correction)
%
%   referenceImage  reference image for error comparison (capital \Delta error)
%                   (optional)
% OUT
%   out                 Structure that contains output data
%     .imagesSC           Single-coil images
%     .imageComb          Combined reconstructed image (cropped to FOV)
%     .imageComb_full     Combined reconstructed image (from oversampled k-space grid)
%     .sens_os            Sensitivity maps
%     .imagesIterSteps    Intermediate images from all CG iterations
%
timerTotal = tic;

if nargin < 3
    referenceImage = [];
end

%% Load input
% Data
signal = data.signal;
k      = data.k_scaled;
nCoils = data.nCoils;

% Properties
image_dim            = properties.image_dim;
do_sense_recon       = properties.do_sense_recon;
undersampling_factor = properties.undersampling_factor;
nIterations          = properties.n_iterations;
oversampling_factor  = properties.gridding.oversampling_factor;
kernel_width         = properties.gridding.kernel_width;
visualization_level  = properties.visualization_level;

%% Adjust signal and k according to SENSE factor
% Select datapoints according to undersampling factor
if numel(undersampling_factor) > 1 
    % selected interleaves given explicitly
    selection = undersampling_factor; 
else
    % choose every R-th data point (R = undersampling factor);
    selection = 1:undersampling_factor:size(signal,2);
end

if undersampling_factor > size(k,2)
    disp(['Undersampling factor chosen greater than maximal possible undersampling factor (Rmax = ' num2str(size(k,2)) ') - Hence set R = ' num2str(size(k,2)) '.']);
end
signal = signal(:,selection,:);
signal = double(reshape(signal, size(signal,1)*size(signal,2), size(signal, 3)));
k      = k(:,selection,:);
k      = reshape(k, size(k,1)*size(k,2), size(k,3));

%% Set up Gridding
grid_size        = round(oversampling_factor*image_dim);                                        % Grid size including oversampling
if mod(grid_size, 2) == 1
    grid_size = grid_size+1;
end 
griddingOperator = prepareGriddingOperator(k', oversampling_factor, kernel_width, grid_size);   % Get gridding operator
center           = ceil(grid_size/2-image_dim/2+1):ceil(grid_size/2+image_dim/2);               % Indices center chunk
%% SENSE Map
% Load SENSE maps or assume sense maps == 1
if do_sense_recon
    senseMaps = data.sense.data;
    mask      = data.sense.mask;
    
    % Adjust size if sense map was created on another size
    if grid_size <= size(senseMaps,1)
        indsCenterFOV = ceil(size(senseMaps,1)/2-grid_size/2+1):ceil(size(senseMaps,2)/2+grid_size/2);
        senseMapsFOV   = senseMaps(indsCenterFOV, indsCenterFOV, :);
        senseMaps      = senseMapsFOV;
        mask           = round(data.sense.mask(indsCenterFOV, indsCenterFOV));
    else
        senseMapsFOV  = zeros(grid_size, grid_size, size(senseMaps,3));
        indsCenterFOV = ceil(grid_size/2-size(senseMaps,1)/2+1):ceil(grid_size/2+size(senseMaps,2)/2);
        senseMapsFOV(indsCenterFOV, indsCenterFOV, :) = senseMaps;
        senseMaps     = senseMapsFOV;
        mask          = zeros(grid_size, grid_size);
        mask(indsCenterFOV, indsCenterFOV) = round(data.sense.mask);
    end
    
%     if (size(senseMapsFOV,1) ~= grid_size)
%         senseMaps_os = zeros(grid_size, grid_size, size(senseMaps,3));
%         for nCoil = 1:size(senseMaps,3)
%             senseMaps_os(center,center,nCoil) = imresize(senseMaps(:,:,nCoil), [image_dim image_dim]);
%         end
%         mask_os = zeros(grid_size, grid_size);
%         mask_os(center,center) = imresize(data.sense.mask, [image_dim image_dim], 'nearest');
%         senseMaps = senseMaps_os;
%         mask      = mask_os;
%     end
else
    senseMaps = ones(ceil(oversampling_factor*image_dim), ceil(oversampling_factor*image_dim), size(signal,2));
    mask      = ones(ceil(oversampling_factor*image_dim), ceil(oversampling_factor*image_dim));
end

%% Noise Covariance Matrix
if isfield(data, 'sense') && isfield(data.sense, 'noiseCovarianceMatrix')          % Load noise covariance matrix if it exists
    noiseCovarianceMatrix = data.sense.noiseCovarianceMatrix;
    noiseCovarianceMatrix = chol(noiseCovarianceMatrix, 'lower');                               % Compute the Cholesky factorization, s.t. psi = L*L'
    signal = (noiseCovarianceMatrix\signal')';                                     % Apply it to signal and sensitivity maps as in the paper (equations 10, 11)
    senseMaps = (noiseCovarianceMatrix\reshape(senseMaps, [], size(senseMaps,3))')';
    senseMaps = reshape(senseMaps, sqrt(size(senseMaps,1)), [], size(senseMaps,2));
end

%% Calculate k-space density for density compensation
CartKspace = griddingOperator.H'*(ones(size(signal(:,1))));     % Grid trajectory
CartKspace = reshape(CartKspace,[grid_size,grid_size]);         % Reshape it to be a square matrix
density    = griddingOperator.H*CartKspace(:);                  % Compute k-space density filter from it

%% Compute k-space filter
% Note: The k-space filter which is computed here is a disk. This k-space
% filter works ONLY for radial and spiral data but NOT for Cartesian
% trajectories. In this case one would have to calculate a square
% k-space filter.
[xx, yy] = meshgrid(linspace(-grid_size/2,grid_size/2,grid_size), linspace(-grid_size/2,grid_size/2,grid_size));
kspFilter = zeros(grid_size,grid_size);
kspFilter(xx.^2 + yy.^2 < (grid_size/2)^2) = 1;
% No k-space filter in FFT of algo.
griddingOperator.kspFilter = ones(size(reshape(kspFilter, [], 1)));

%% Visualize k-space filter
if properties.visualization_level > 1
    stringTitle = 'k-Space Filter';
    figure('Name', stringTitle);
    imagesc(abs(kspFilter));
    axis square;
    colormap gray
    title(stringTitle);
end

%% Calculate root of sum of squares coil intensity for intentsity correction
intensity = sqrt((sum(conj(senseMaps).*senseMaps, 3))+1E-15);

%% Visualize intensity correction
if properties.visualization_level > 1 && properties.do_sense_recon
    stringTitle = 'Bias Field for Intensity Correction';
    figure('Name', stringTitle);
    imagesc(abs(intensity));
    axis square;
    colormap gray
    title(stringTitle);
    
    %% Visualize Sensitivity maps
    stringTitle = 'Sense maps (abs)';
    figure('Name', stringTitle);
    montage(abs(permute(senseMaps, [1 2 4 3])));
    axis square;
    colormap gray
    title(stringTitle);
    
    stringTitle = 'Sense maps (angle)';
    figure('Name', stringTitle);
    montage(angle(permute(senseMaps, [1 2 4 3])), 'DisplayRange', [-pi,pi]);
    axis square;
    colormap gray
    title(stringTitle);
end

%% Image reconstruction using iterative CG reconstruction
imagesIterSteps     = cell(nIterations+1,1);        % Array for images in all iteration steps
durationIterSteps   = zeros(nIterations+1,1);       % stores runtime per iteration
Deltas              = zeros(nIterations+1,1);       % Delta variable (current image - final image or reference if given)
deltas              = zeros(nIterations+1,1);       % delte (residuum of current solution, i.e. norm(E*current_image - signal)
% Perform CG-SENSE algorithm
coilsSelect = 1:nCoils;
density = repmat(density, 1, size(signal,2));

%% initialization starting values and CG vectors ("Zeroth Iteration")
timerIteration = tic;
a = EH( signal(:,coilsSelect)./density, senseMaps(:,:,coilsSelect), griddingOperator)./intensity;

if ~isempty(referenceImage)
    Deltas(1) = norm(reshape( referenceImage.mask.* ((a(center,center)) - referenceImage.image ),1,[]))./ norm(reshape(referenceImage.mask.*referenceImage.image, 1, []));
end

reconImage = mask.*a./intensity;
b = zeros([grid_size grid_size]);
p = a;
r = a;
imagesIterSteps{1} = transformKspaceToImage(transformImageToKspace(reconImage).*...
kspFilter);
imagesIterSteps{1} = imagesIterSteps{1}(center,center);
durationIterSteps(1) = toc(timerIteration);

%% CG Loop
for iIteration = 1:nIterations
    
    timerIteration = tic;
    
    if properties.visualization_level > 0
        fprintf('\tIteration %d...', iIteration)
    end
    
    deltas(iIteration) = r(:)'*r(:)/(a(:)'*a(:));
    q = EH( E(p./intensity , senseMaps(:,:,coilsSelect), griddingOperator)./density, senseMaps(:,:,coilsSelect), griddingOperator)./intensity;
    b = b + r(:)'*r(:)/(p(:)'*q(:))*p;
    r_new = r - r(:)'*r(:)/(p(:)'*q(:))*q;
    p = r_new + r_new(:)'*r_new(:)/(r(:)'*r(:))*p;
    r = r_new;
    previousReconImage = reconImage; % for difference plots
    reconImage = mask.*b./intensity;
    
    if visualization_level > 0
        plotIteration(reconImage, deltas, center, iIteration, ...
            previousReconImage);
    end
    
    imagesIterSteps{iIteration+1} = transformKspaceToImage(transformImageToKspace(reconImage).*...
    kspFilter);
    imagesIterSteps{iIteration+1} = imagesIterSteps{iIteration+1}(center,center);
    
    if ~isempty(referenceImage)
        Deltas(iIteration+1) = norm(reshape( referenceImage.mask.* ((reconImage(center,center)) - referenceImage.image ),1,[]))./ norm(reshape(referenceImage.mask.*referenceImage.image, 1, []));
    end
    
    durationIterSteps(iIteration+1) = toc(timerIteration);

    if properties.visualization_level > 0
        fprintf('%.3f s\n', durationIterSteps(iIteration+1))
    end
    
end
deltas(nIterations+1) = r(:)'*r(:)/(a(:)'*a(:));

%% Output
% Final application of k-space filter
reconImage = transformKspaceToImage(transformImageToKspace(reconImage).*...
    kspFilter);
reconImage = reconImage(center,center);

% Assemble Output
out.imageComb = reconImage;
out.sens_os = senseMaps;
out.kspFilter = kspFilter;
out.center = center;
out.deltas = deltas;
out.imagesIterSteps = imagesIterSteps;
if ~isempty(referenceImage)
    out.Deltas = Deltas;
end

out.durationIterSteps = durationIterSteps;
out.totalElapsedTime = toc(timerTotal);

end
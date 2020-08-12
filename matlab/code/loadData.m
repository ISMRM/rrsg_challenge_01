function data = loadData(datafile)

% Function to load datasets from a specified folder
% Content of a h5 file can be easily shown by the command:
% h5disp(datafile)
% The function aims to load the following datasets
% - Coils
% - noise
% - rawdata
% - trajectory
col = @(x) x(:);
% Load signal data (coils | interleaves/spokes | samples)
rawdataStruct = h5read(datafile,'/rawdata');
signal        = double(rawdataStruct.r+1i*rawdataStruct.i); clear rawdataStruct;
[nCoils,nInterleaves,nSamples] = size(signal);
signal = permute(signal, [3 2 1]);

% Load trajectory (interleaves | samples | coordinate)
k = h5read(datafile,'/trajectory');
k = permute(k, [2 1 3]);
% Some datasets hat an (empty) 3rd dimension which is not required in this
% 2D reconstruction
if size(k,3) == 3
    k(:,:,3) = [];
end
% Norm k to range from -0.5 to +0.5
% Center out trajectory check
center_out_scale = 1;
if all(col(k(1,:,:)) == 0)
   % Seems to be measured center out (e.g. spiral)
   center_out_scale = 2;
end
range           = center_out_scale*...
    max(sqrt(sum((k(end,:,:) - k(1,:,:)).^2,3)));
% k_scaled = k/range;
% k_scaled(:,:,1) = k_scaled(:,:,1) - min(min(k_scaled(:,:,1))) - 0.5;
% k_scaled(:,:,2) = k_scaled(:,:,2) - min(min(k_scaled(:,:,2))) - 0.5;

% Load SENSE maps
senseMaps = h5read(datafile,'/Coils');
senseMaps = double(senseMaps.r+1i*senseMaps.i);
senseMaps = fliplr(rot90(senseMaps,3));
% Compute image resolution and overgrid factor from k-space data if given in units 1/FOV
Nimg = double(floor(round(range,4)));
if mod(Nimg,2)
   % Pad to symmetric resolution
   Nimg = Nimg+1;
end

overgrid_factor_a = 1/norm(...
    squeeze(k(end-1, 1, :)-k(end, 1, :)));
overgrid_factor_b = 1/norm(...
    squeeze(k(1, 1, :)-k(2, 1, :)));
overgrid_factor = double(min(overgrid_factor_a, overgrid_factor_b));

center = ceil(size(senseMaps,1)/2-Nimg/2)+1:ceil(size(senseMaps,1)/2+Nimg/2);
senseMapsResized = zeros(Nimg, Nimg, size(senseMaps,3));
for nCoil = 1:size(senseMaps,3)
    senseMapsResized(:,:,nCoil) = senseMaps(center,center,nCoil);
end
senseMaps = senseMapsResized;
% Otherwise one could take the resolution of the sense maps
Nimg      = size(senseMaps,1);
% Compute Mask
% sumMagSenseMaps = sum(abs(senseMaps),3);
% mask      = zeros(Nimg, Nimg);
% mask(sumMagSenseMaps > 0) = 1;
try mask = h5read(datafile,'/mask').';
  mask = mask(center, center);
catch
  warning('No mask in provided sense.h5 file.');
  mask = ones(Nimg, Nimg);
end
% It was noted that too large masks can cause high intensity voxels on the
% border of the mask - hence, the mask is reduced a bit exploiting
% knowledge on the mean magnitude of the sense maps (0.2 chosen
% arbitrarily)
% meanMagSenseMaps = mean(sumMagSenseMaps(logical(mask)));
% mask      = zeros(Nimg, Nimg);
% mask(sumMagSenseMaps > .3*meanMagSenseMaps) = 1;

% Load noise covariance matrix
try h5read(datafile,'/noise');
    noiseCovarianceMatrix      = h5read(datafile,'/noise');
    noiseCovarianceMatrix      = double(noiseCovarianceMatrix.r+1i*noiseCovarianceMatrix.i);
    data.sense.noiseCovarianceMatrix = noiseCovarianceMatrix;
catch
    warning('No noise covariance matrix in provided sense.h5 file.');
end

% -- Output
data.signal     = signal;
data.k          = double(k);
data.k_scaled   = double(k);
data.nCoils     = nCoils;
data.nFE        = nSamples;
data.nSpokes    = nInterleaves;
data.Nimg       = Nimg;
data.sense.data = senseMaps;
data.sense.mask = mask;
data.overgrid_factor = overgrid_factor;
end
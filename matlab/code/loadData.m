function data = loadData(pathDataset)

% Function to load datasets from a specified folder
% The function aims to load the following files
% - data.h5 containing signal data and trajectory
% - senseMaps.h5 containing senseMaps
% - noiseCovarianceMatrix.h5 containing the noise covariance matrix

% -- Load from datafile
datafile = fullfile(pathDataset,'data.h5');

% Load signal data
rawdata_real = h5read(datafile,'/rawdata');
rawdata      = double(rawdata_real.r+1i*rawdata_real.i); clear rawdata_real;
signal       = rawdata;
[nFE,nSpokes,nCoils] = size(signal);

% Load trajectory
trajectory  = h5read(datafile,'/trajectory');
k           = trajectory;
% Norm k to range from -0.5 to +0.5
k_scaled        = zeros(size(k));
range           = max(reshape(k, [], 2))-min(reshape(k, [], 2));
k_scaled(:,:,1) = k(:,:,1)/range(1);
k_scaled(:,:,2) = k(:,:,2)/range(2);
k_scaled(:,:,1) = k_scaled(:,:,1) - min(min(k_scaled(:,:,1))) - 0.5;
k_scaled(:,:,2) = k_scaled(:,:,2) - min(min(k_scaled(:,:,2))) - 0.5;

% Load image resolution from h5 file as well, otherwise resolution of SENSE
% maps is used as recon size
try h5read(datafile,'/Nimg');
    Nimg = h5read(datafile,'/Nimg');
end

% Load SENSE maps
datafile  = fullfile(pathDataset,'sense.h5');
senseMaps = h5read(datafile,'/senseMaps');
senseMaps = double(senseMaps.r+1i*senseMaps.i);
if ~exist('Nimg', 'var')
    Nimg  = size(senseMaps,1);
end
% Load Mask
mask = h5read(datafile,'/mask');
% Load noise Covariance Matrix
try h5read(datafile,'/senseMaps');
    noiseCovarianceMatrix = h5read(datafile,'/noiseCovarianceMatrix');
    noiseCovarianceMatrix = double(noiseCovarianceMatrix.r+1i*noiseCovarianceMatrix.i);
    data.sense.noiseCovarianceMatrix = noiseCovarianceMatrix;
catch
    warning('No noise covariance matrix in provided sense.h5 file.');
end

% -- Output
data.signal    = signal;
data.k         = k;
data.k_scaled  = k_scaled;
data.nCoils    = nCoils;
data.nFE       = nFE;
data.nSpokes   = nSpokes;
data.Nimg      = Nimg;
data.sense.data = senseMaps;
data.sense.mask = mask;

end
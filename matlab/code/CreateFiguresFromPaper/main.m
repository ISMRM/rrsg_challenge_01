% This script loads the provided data, loads (or creates) SENSE maps and
% reconstructs the data for different undersampling factors (R values).
% Reconstructions are done in the CreateFigure functions which are meant to
% replicate the Figures of the original paper.

%% Setup Data and Results paths
% NOTE: Adapt data paths to folder containing the h5 files
pathDataBrain   = '../../../../CGSENSE_challenge_sub/data/rawdata_brain_radial_96proj_12ch.h5';
pathDataHeart   = '../../../../CGSENSE_challenge_sub/data/rawdata_heart_radial_55proj_34ch.h5';
pathResults     = 'results'; % path where results figures shall be stored

% create path for results
[~,~] = mkdir(pathResults);

%% Create Figure 4 (brain data)
dataBrain                           = loadData(pathDataBrain);
% [Deltas, deltas, durationIterSteps] = createFigure4(dataBrain, pathResults);

%% Create Figure 5 (brain data)
createFigure5(dataBrain, pathResults);

%% Create Figure 6 (heart data)
dataHeart                   = loadData(pathDataHeart);
createFigure6(dataHeart, pathResults);
function fh = createFigure5(data, pathResults)
% Reproduces Figure 5 of paper Pruessmann et al, 2001
%
%   fh = createFigure5(data)
%
% IN
%
% OUT
%
% EXAMPLE
%   createFigure5
%
%   See also CGSense

% Author:   Franz Patzig, Thomas Ulrich, Maria Engel, Lars Kasper
% Created:  2019-04-30
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
%
iCoil = 1;

stringTitle = sprintf('Figure 5 Reproduction - Undersampled Recons Arbitrary SENSE');
fh(1) = figure('Name', [stringTitle, ' - part 1']);
fh(2) = figure('Name', [stringTitle, ' - part 2']);

RArray = [1 2 3 4];
nR = numel(RArray);
dataIn = data;

data.Nimg = 300;
properties.image_dim = data.Nimg;             % For the phantom it should be 152, 340 for brain, 360 for heart
properties.gridding.oversampling_factor = 1.7033398310591292;  % Gridding oversampling factor
properties.gridding.kernel_width = 4;         % Gridding kernel width as a multiple of dk without oversampling
properties.visualization_level = 1;

%% Compute single coil recons w/o SENSE (left column of plot) and SENSE recons%
outSense  = cell(nR,1);
outSingle = cell(nR,1);
for iR = 1:nR
    fprintf('Reconstruct Image with R = %d... (%d/%d)\n', RArray(iR), iR, nR);
    dataTmp = dataIn;
    properties.undersampling_factor = iR;          % Undersampling factor (positive integer)
    
    % SENSE reconstruction
    properties.n_iterations = 10;
    properties.do_sense_recon = 1;
    out = CGSense(dataTmp, properties);
    outSense{iR} = out;
    
    % Single coil FFT, simplify some parameters and strip data to selected
    % coil
    properties.n_iterations = 1;
    properties.do_sense_recon = 0;
    dataTmp.sense.data = dataTmp.sense.data(:,:,iCoil);
    dataTmp.sense.noiseCovarianceMatrix = 1; % ignore noise covariance
    dataTmp.signal = dataTmp.signal(:,:,iCoil);
    dataTmp.nCoils = 1;
    out = CGSense(dataTmp, properties);
    outSingle{iR} = out;
end
save('result_brain.mat','outSingle','outSense');
%% Create Subplots for figure
bestIteration = [5 5 5 5];
for iR = 1:nR
    R = RArray(iR);
    
    if iR < 3
        figure(fh(1))
    else
        figure(fh(2));
    end
    
    for iCol = 1:3
        
        switch iCol
            case 1
                out = outSingle{iR}.imageComb;
            case 2
                out = outSense{iR}.imagesIterSteps{1,1};
            case 3
                out = outSense{iR}.imagesIterSteps{bestIteration(iR),1};
        end
        
        subplot(2,3,3*(2-mod(iR,2)-1)+iCol);
        I = abs(out);
        imagesc(I);
        axis square
        %axis off
        axis image
        colormap gray
        % caxis([0, intensityMaxPerColumn(iCol)]);
        switch iCol
            case 1
                ylabel(sprintf('R = %d', R));
                title(sprintf('single coil recon (coil %d)', iCoil));
            case 2
                title('Initial (E^H)');
            case 3
                title(sprintf('Final (%d)', bestIteration(iR)));
        end
        set(gca, 'XTick', [])
        set(gca, 'YTick', [])
    end
    
    
end

%% save figure
for iPart = 1:2
    print(fh(iPart),[pathResults '/Figure5_undersamplingRecon_part' ...
        num2str(iPart)],'-dpng')
end
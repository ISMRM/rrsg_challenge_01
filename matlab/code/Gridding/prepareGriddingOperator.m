function [ griddingOp ] = prepareGriddingOperator(k, os, width, N)
% Prepare the gradient operator
% Input:
% os:    (scalar) oversampling factor
% width: (scalar) kernel width
% N:     (scalar) size of oversampled grid

    % Set kernel attributes
    ns    = 10000*width/2;  % Nr of samples for Kernel-calculation per kernel point.
    beta  = pi*sqrt(width^2/os^2*(os-0.5)^2-0.8);	% beta by Beatty et al.
    vals  = kaiser(2*ns,beta);  vals(1:ns) = [];

    % Setup the kernel structure for the convolution kernel:
    kernel.values = vals;
    kernel.kwidth = width/2;%*os;

    % Normalize gridding kernel
%     gDirac = gridFunction(1,[0;0],N,kernel,true);
%     kernel.values = kernel.values/sum(gDirac(:));

    % Derive the Apodization Correction Matrix
    W = abs(ifftshift(ifft2(ifftshift(gridFunction(1,[0;0],N,kernel,true)))));
    % Contribution from nearest neighbour interpolation of gridding kernel
    h = squeeze(sinc(linspace(-N/2, N/2-1, N)/(N*ns)));
    W = W.*(h'*h);
    W = W./max(W(:));
%     W = W./os;
%     W(W<1e-10) = 1e-10;
    H = gridOperator(k*os,N,kernel);
    griddingOp.H = H;
    griddingOp.W = W;
    griddingOp.N_samples = size(k,2);
    griddingOp.N_grid = N;
    
end


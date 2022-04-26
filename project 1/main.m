clear 
clf
close all

%Load titan images in grayscale 0-1 format and plot
img_titan = double(imread("titan.jpg"))/255;
img_rosetta = double(rgb2gray(imread("rosetta.jpg")))/255;

figure
subplot(1,2,1)
imagesc(img_titan)
title('Titan')

subplot(1,2,2)
imagesc(img_rosetta)
title('Rosetta')
colormap 'gray'

p_c = 0.5;
sz = 10000;
reconstruction(img_titan, p_c, sz, 'Titan')
reconstruction(img_rosetta, p_c, sz, 'Rosetta')

%% FUNCTIONS
function reconstruction(img, p_c, sz, img_name)

    if nargin < 4
        img_name = '';
    end
    %Generate data of observed points and ones to predict
    [x_o, x_m, ind_o, ind_m] = generate_data(img, p_c);
    
    %Generate mesh of all possible locations
    [m, n] = size(img);
    [X, Y] = meshgrid(1:m, 1:n);
    loc = [X(:), Y(:)];

    %If subset size is larger than amount of observed pixels, correct:
    if sz > length(x_o)
        sz = length(x_o);
    end

    %Get observed and missing locations:
    loc_o = loc(ind_o, :);
    loc_m = loc(ind_m, :);
    
    %Covariates (intercept + horizontal + vertical)
    %Observed pixels: 
    B0_o = ones(size(x_o));
    BX_o = loc_o(:,1);
    BY_o = loc_o(:,2);
    cov_o = [B0_o BX_o BY_o];
    
    %Missing pixels:
    B0_m = ones(size(x_m));
    BX_m = loc_m(:,1);
    BY_m = loc_m(:,2);
    cov_m = [B0_m BX_m BY_m];
    
    %Least squares estimate and means
    LSE = (cov_o'*cov_o)\cov_o'*x_o;
    mu_o = cov_o*LSE;
    mu_m = cov_m*LSE;
    
    %Residuals of observed pixels:
    e_o = x_o(1:sz) - mu_o(1:sz);
    
    %Fit variogram to residual of observed pixels to estimate parameters
    n_bins = 100;
    emp_v = emp_variogram(loc_o(1:sz, :), e_o, n_bins);
    lse = cov_ls_est(e_o, 'matern', emp_v, struct('nu', 1));

    %Plot variogram
    plot_variogram(lse, emp_v, img_name)
    
    %% GMRFs
    
    %Stencils:
    q1 = [0 0 0 0 0; 
          0 0 0 0 0;
          0 0 1 0 0;
          0 0 0 0 0;
          0 0 0 0 0];
    
    q2 = [0 0 0 0 0;
          0 0 -1 0 0;
          0 -1 4 -1 0;
          0 0 -1 0 0;
          0 0 0 0 0];
    
    q3 = [0 0 1 0 0;
          0 2 -8 2 0;
          1 -8 20 -8 1;
          0 2 -8 2 0;
          0 0 1 0 0];
    
    tau = 2*pi/lse.sigma^2;
    Q = @(kappa) tau*stencil2prec([m, n], kappa^4*q1 + 2*kappa^2*q2 + q3);
    
    %Generate reconstructed image with estimated kappa
    Recon_image = krig_img(Q, lse.kappa, ind_o, ind_m, mu_o, mu_m, x_o);
    
    %Plot reconstructed image, original image and differences
    figure
    subplot(1,3,1)
    imagesc(reshape(Recon_image,[m,n]));
    title("Reconstructed Image");
    axis image
    
    subplot(1,3,2)
    imagesc(img);
    title("Original Image")
    axis image
    
    subplot(1,3,3)
    imagesc(reshape(Recon_image,[m,n]) - img);
    title("Differences")
    axis image
    sgtitle(img_name)
    colormap 'gray'
end

function plot_variogram(lse, emp_v, img_name)
    %Plot fitted matern variogram from the binned variogram estimate
    
    %Matern variogram to estimated parameters
    mat_v = matern_variogram(emp_v.h, lse.sigma, lse.kappa, lse.nu, lse.sigma_e);

    figure
    plot(emp_v.h, mat_v)
    hold on
    plot(emp_v.h, emp_v.variogram, '.')
    title(img_name, 'Estimated matern variogram & Binned variogram estimate')
    legend('Matern variogram', 'Binned variogram estimate')
    hold off
end

function Recon_image = krig_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o)
    %Use estimated parameters to generate GMRF:
    Q = Q(kappa);

    %Precision matrices for observed and missing indices
    Qom = Q(ind_o, ind_m);
    Qm = Q(ind_m, ind_m);

    %Make kriging estimate
    krig = mu_m - Qm\(Qom'*(x_o - mu_o));

    Recon_image = zeros(length(ind_o) + length(ind_m), 1);
    Recon_image(ind_o) = x_o;
    Recon_image(ind_m) = krig;
end

function [x_o, x_m, ind_o, ind_m] = generate_data(img, p)
% GENERATE_DATA     Create independent sample of observed pixels from
% grayscale image IMG, given probability P
    [m, n] = size(img);
    N_o = round(p*m*n);

    ind = randperm(m*n)';
    ind_o = ind(1:N_o);
    ind_m = ind(N_o+1:end);

    x_o = img(ind_o);
    x_m = img(ind_m);
end
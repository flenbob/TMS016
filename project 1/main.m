%Load titan images in grayscale 0-1 format and plot
clear
img_titan = double(imread("titan.jpg"))/255;
img_rosetta = double(imread("rosetta.jpg"))/255;

figure(1)
subplot(1,2,1)
imagesc(img_titan)
title('Titan')

subplot(1,2,2)
imagesc(img_rosetta)
title('Rosetta')
colormap 'gray'

%Generate data of observed points and ones to predict
p_c = 0.49;
[x_o, x_m, ind_o, ind_m] = generate_data(img_titan, p_c);

%Generate mesh of all possible locations
[m, n] = size(img_titan);
[X, Y] = meshgrid(1:m, 1:n);
loc = [X(:), Y(:)];

%Get observed and missing locations:
loc_o = loc(ind_o, :);
loc_m = loc(ind_m, :);

%Parameters for gauss field with matern covariance
sz = 2000; %set size for faster computation
nu = 1;
sigma = 1;
kappa = 1;

Xs = generate_gauss_field(nu, sigma, kappa, loc(1:sz, :));

%Covariates (intercept + horizontal + vertical)
%Observed values: 
B0_o = ones(size(x_o));
BX_o = loc_o(:,1);
BY_o = loc_o(:,2);
cov_o = [B0_o BX_o BY_o];

%Missing values:
B0_m = ones(size(x_m));
BX_m = loc_m(:,1);
BY_m = loc_m(:,2);
cov_m = [B0_m BX_m BY_m];

%Least squares estimate
LSE = (cov_o'*cov_o)\cov_o'*x_o;
mu_o = cov_o*LSE;
mu_m = cov_m*LSE;

%Predicted values for missing pixels
Y_hat = mu_m(1:sz) + Xs;

%Get residuals (observed vs predicted values):
e = x_o(1:sz) - Y_hat;

%Bin residuals and fit variogram
h = linspace(0, n, 100);
mat_v = matern_variogram(h, sigma, kappa, nu, 0); %True vs binned estimate of the variogram

%Estimate variogram
emp_v = emp_variogram(loc(1:sz, :), e, 500);

lse = cov_ls_est(e, 'matern', emp_v);

mat_v_new = matern_variogram(h, lse.sigma, lse.kappa, ...
                            lse.nu, lse.sigma_e);

figure(2)
plot(h,mat_v)
hold on
plot(emp_v.h,emp_v.variogram,'.')
plot(h,mat_v_new)
hold off

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

%Optimize kappa value. Initialize parameters:
Recon_image = recon_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o);
prev_error = sum(abs(Recon_image - reshape(img_titan, [m*n, 1])));
error = 0;

kappa = lse.kappa;
alpha = 3; %step scaling factor
h = 1e-2; %step to determine derivative
img_array = reshape(img_titan, [m*n, 1]); 

while prev_error > error 
    %Get two image estimates with 'kappa' and 'kappa+h' to determine
    %descent direction
    Recon_image = recon_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o);
    Recon_image_h = recon_img(Q, kappa+h, ind_o, ind_m, mu_o, mu_m, x_o);

    %Differences 
    diff = sum(abs(Recon_image - img_array));
    diff_h = sum(abs(Recon_image_h - img_array));
    
    %Derivative
    df = (diff_h-diff)/h;

    %Update new kappa and save previous in case of optimality:
    kappa_prev = kappa
    kappa = kappa - alpha/df;

    %Save error for next iteration:
    prev_error = sum(abs(Recon_image - reshape(img_titan, [m*n, 1])))

    %Calculate error from new kappa:
    Recon_image = recon_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o);
    error = sum(abs(Recon_image - reshape(img_titan, [m*n, 1])));
end

%Generate reconstructed image with 'optimal' kappa
kappa = kappa_prev;
Recon_image = recon_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o);

%Plot reconstructed image, original image and differences
figure(3)
subplot(1,3,1)
imagesc(reshape(Recon_image,[m,n]));
title("Reconstructed Image");
axis image

subplot(1,3,2)
imagesc(img_titan);
title("Original Image")
axis image

subplot(1,3,3)
imagesc(reshape(Recon_image,[m,n]) - img_titan);
title("Differences")
axis image
colormap 'gray'

%% FUNCTIONS
function Recon_image = recon_img(Q, kappa, ind_o, ind_m, mu_o, mu_m, x_o)
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

function Z = generate_gauss_field(nu, sigma, kappa, loc)
    %Given location matrix and parameters, generate a 
    %Gaussian field with a Matérn covariance
    D = squareform(pdist(loc));
    ma_co = matern_covariance(D, sigma, kappa, nu);
    R = chol(ma_co);
    
    %Randomize normally distributed z ~ N(0,1)
    z = randn(size(loc, 1), 1);
    
    %Return zero mean Gaussian random field:
    Z = R*z;
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
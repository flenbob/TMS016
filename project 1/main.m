%Load titan images in grayscale 0-1 format and plot
clear
img_titan = double(imread("titan.jpg"))/255;
img_rosetta = double(imread("rosetta.jpg"))/255;

figure(1)
subplot(1,2,1)
imshow(img_titan)
title('Titan')

subplot(1,2,2)
imshow(img_rosetta)
title('Rosetta')

%Generate data of observed points and ones to predict
p_c = 0.5;
[x_o, x_c] = generate_data(img_titan, p_c);

[m, n] = size(img_titan);
[X, Y] = meshgrid(1:m, 1:n);
loc = [X(:), Y(:)];

%Subset size for faster computation:
sz = 1000;
D = squareform(pdist(loc(1:sz, :))); %reduce to first 10k for comp. time

%Matern covariance
nu = 1;
sigma = 1;
kappa = 5;
ma_co = matern_covariance(D, sigma, kappa, nu);
R = chol(ma_co);

%Randomize normally distributed z ~ N(0,1)
z = randn(size(loc(1:sz), 2), 1);

%Mean zero Gaussian random field:
X = R*z;



function [x_o, x_m, ind_o, ind_m] = generate_data(img, p)
% GENERATE_DATA     Create independent sample of observed pixels from
% grayscale image IMG, given probability P
    [m, n] = size(img);
    N_o = round(p*m*n);

    ind = randperm(m*n);
    ind_o = ind(1:N_o);
    ind_m = ind(N_o+1:end);

    x_o = img(ind_o);
    x_m = img(ind_m);
end
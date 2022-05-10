clear
close all
load('permeability.mat')

k = 2;
%Plot without noise
segment_image(Y, k, 0, "no noise");

%Plot with noise
scale = 1;
sigma_e = scale*randn(size(Y, 1)*size(Y, 2), 1);

segment_image(Y, k, sigma_e, "1 Sigma noise")

segment_image(Y, k, 3*sigma_e, "3 Sigma noise")

function segment_image(img, k, sigma_e, plot_title)
    %Segment image into k classes and plot result for: 
    % - Gaussian mixture model (gmm)
    % - K-means (km)
    % - Markov Random field model (mrf)
    %, with a plot title passed to the function
    [m, n] = size(img);
    X = reshape(img, [m*n, 1]);
    X = X + sigma_e; %add noise to image

    %K-means
    labels_km = kmeans(X, k);
    
    %Gaussian mixture
    [params_gmm, ~] = normmix_sgd(X, k);
    [labels_gmm, ~] = normmix_classify(X, params_gmm);
    
    %MRF mixture
    [~, ~, ~, label_mrf, ~] = mrf_sgd(reshape(X, [m, n]), k);

    %Plot all model label images
    figure
    subplot(3,1,1)
    imagesc(reshape(labels_km, [m n]))
    title('K-means labeling')
    
    subplot(3,1,2)
    imagesc(reshape(labels_gmm, [m n]))
    title('Gaussian mixture labeling')
    
    subplot(3,1,3)
    imagesc(label_mrf)
    title('Markov random field labeling')
    sgtitle(plot_title)
    colormap 'gray'
end
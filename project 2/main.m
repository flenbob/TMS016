clear
close all
load('permeability.mat')

%Show image of horizontal permeability slice
figure(1)
imagesc(Y)
colormap 'gray'

%Reshape image
[m, n] = size(Y);
X = reshape(Y, [n*m 1]);

%K-means
k = 2;
labels_km = kmeans(X, k);

%Gaussian mixture
[params_gmm, ~] = normmix_sgd(X, k);
[labels_gmm, p_gmm] = normmix_classify(X, params_gmm);

%Markov RF mixture
[theta, alpha, beta, cl, p] = mrf_sgd(Y,k);

%Neighborhood stencil and parameters
N = [0 1 0;
     1 0 1;
     0 1 0];
beta = beta*eye(k);

%Set a starting value
x = randi(k,[m n]); 
z0 = zeros(m,n,k);
for i=1:m
    for j=1:n
        z0(i,j, x(i,j)) = 1;
    end
end

%Simulate z using Gibbs sampling
[z, Mz,ll] = mrf_sim(z0,N,alpha,beta,100,0);
labels_mrf = zeros(m,n);

for i=1:k
    labels_mrf = labels_mrf + i*(z(:,:,i) == 1);
end

%Plot all model label images
subplot(3,1,1)
imagesc(reshape(labels_km, [m n]))
title('K-means labeling')

subplot(3,1,2)
imagesc(reshape(labels_gmm, [m n]))
title('Gaussian mixture labeling')

subplot(3,1,3)
imagesc(reshape(labels_mrf, [m n]))
title('Markov random field labeling')
colormap 'gray'
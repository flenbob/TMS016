%Load titan images in grayscale 0-1 format and plot
clear
img = double(imread("titan.jpg"))/255;
img2 = double(imread("titan2.jpg"))/255;

figure(1)
sgtitle('Titan images')
subplot(1,2,1)
imshow(img)
subplot(1,2,2)
imshow(img2)

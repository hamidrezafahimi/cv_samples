clear;
clc;

I=imread('fig2.jpg');
J= rgb2gray(I);
figure(2)
imshow(J);
[glcm1,SI1] = graycomatrix(J,'Offset',[0 1],'GrayLimits', [0 255], 'NumLevels', 4,'Symmetric',true);
Stats = graycoprops(glcm1,{'contrast','homogeneity','Correlation','Energy'});
disp(Stats);
figure(3)
imshow(rescale(SI1));

[glcm2,SI2] = graycomatrix(J,'Offset',[-1 1],'GrayLimits', [0 255], 'NumLevels', 4,'Symmetric',true);
Stats = graycoprops(glcm2,{'contrast','homogeneity','Correlation','Energy'});
disp(Stats);
figure(4)
imshow(rescale(SI2));

[glcm3,SI3] = graycomatrix(J,'Offset',[-1 0],'GrayLimits', [0 255], 'NumLevels', 4,'Symmetric',true);
Stats = graycoprops(glcm3,{'contrast','homogeneity','Correlation','Energy'});
disp(Stats);
figure(5)
imshow(rescale(SI3));

[glcm4,SI4] = graycomatrix(J,'Offset',[-1 -1],'GrayLimits', [0 255], 'NumLevels', 4,'Symmetric',true);
Stats = graycoprops(glcm4,{'contrast','homogeneity','Correlation','Energy'});
disp(Stats);
figure(6)
imshow(rescale(SI4));




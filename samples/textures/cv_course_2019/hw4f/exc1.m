% GLCM feature extraction

clear;


offset_GLCM = [0 1; -1 1];
offset = [1*offset_GLCM ; 2*offset_GLCM];
img = rgb2gray(imread('fig3.jpg'));

[Grauwertmatrix, SS] = graycomatrix(img,'NumLevels', 6, 'GrayLimits', [], 'Offset',offset);
GrauwertStats = graycoprops(Grauwertmatrix, {'contrast','homogeneity','Correlation','Entropy'});
% GLCMFeatureVector = [mean(GrauwertStats.Contrast) mean(GrauwertStats.Correlation) mean(GrauwertStats.Energy) mean(GrauwertStats.Homogeneity)];
imshow(rescale(SS));
disp(GrauwertStats);

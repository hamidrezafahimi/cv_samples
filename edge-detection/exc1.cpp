#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
Mat image;
//-------------------------------------------------------------------------------
//part: 1-1:
//-------------------------------------------------------------------------------
/*image = imread("Fig1.jpg");
Mat gblurim, laplaceresult1, laplaceresult3, laplaceresult5 ;
GaussianBlur(image, gblurim, Size(3,3),1.5);

Laplacian(gblurim,laplaceresult1, CV_8U, 1);
imshow("Laplacian of Gussian 1",laplaceresult1);
imwrite("LoG1.jpg", laplaceresult1);

Laplacian(gblurim,laplaceresult3, CV_8U, 3);
imshow("Laplacian of Gussian 3",laplaceresult3);
imwrite("LoG3.jpg", laplaceresult3);

Laplacian(gblurim,laplaceresult5, CV_8U, 5);
imshow("Laplacian of Gussian 5",laplaceresult5);
imwrite("LoG5.jpg", laplaceresult5);

//-------------------------------------------------------------------------------
//part: 1-2:
//-------------------------------------------------------------------------------
Mat gradx3, grady3, gradx5, grady5, gradx7, grady7;

image = imread("Fig1.jpg");
Sobel(image, gradx3, CV_8U, 1, 0, 3);
Sobel(image, grady3, CV_8U, 0, 1, 3);
imshow("Result of Sobel x kernelsize 3", gradx3);
imshow("Result of Sobel y kernelsize 3", grady3);
imwrite("gradx3.jpg",gradx3);
imwrite("grady3.jpg",grady3);

image = imread("Fig1.jpg");
Sobel(image, gradx5, CV_8U, 1, 0, 5);
Sobel(image, grady5, CV_8U, 0, 1, 5);
imshow("Result of Sobel x kernelsize 5", gradx5);
imshow("Result of Sobel y kernelsize 5", grady5);
imwrite("gradx5.jpg",gradx5);
imwrite("grady5.jpg",grady5);

image = imread("Fig1.jpg");
Sobel(image, gradx7, CV_8U, 1, 0, 7);
Sobel(image, grady7, CV_8U, 0, 1, 7);
imshow("Result of Sobel x kernelsize 7", gradx7);
imshow("Result of Sobel y kernelsize 7", grady7);
imwrite("gradx7.jpg",gradx7);
imwrite("grady7.jpg",grady7);

//-------------------------------------------------------------------------------
//part: 1-3:
//-------------------------------------------------------------------------------

Mat gradient3, gradient5, gradient7;

gradient3 = abs(gradx3) + abs(grady3);
imshow("The gradient 3", gradient3);
imwrite("gradient3.jpg",gradient3);

gradient5 = abs(gradx5) + abs(grady5);
imshow("The gradient 5", gradient5);
imwrite("gradient5.jpg",gradient5);

gradient7 = abs(gradx7) + abs(grady7);
imshow("The gradient 7", gradient7);
imwrite("gradient7.jpg",gradient7);

//-------------------------------------------------------------------------------
//part: 1-4:
//-------------------------------------------------------------------------------

Mat cannyimage1, cannyimage2, cannyimage3;
image = imread("Fig1.jpg");
Canny(image, cannyimage1, 250, 250, 3);
Canny(image, cannyimage2, 300, 400, 3);
Canny(image, cannyimage3, 200, 500, 3);
imshow("canny result 1", cannyimage1);
imshow("canny result 2", cannyimage2);
imshow("canny result 3", cannyimage3);
imwrite("cannyimage1.jpg",cannyimage1);
imwrite("cannyimage2.jpg",cannyimage2);
imwrite("cannyimage3.jpg",cannyimage3);
//*/
//-------------------------------------------------------------------------------
//part: 1-5:
//-------------------------------------------------------------------------------
image = imread("Fig1.jpg");
Mat gimage, glaplaceres, ggblurim, ggradx, ggrady, ggradient, gcannyimage;
GaussianBlur(image, gimage, Size(3,3), 1.5);

GaussianBlur(gimage, ggblurim, Size(3,3),1.5);
Laplacian(ggblurim, glaplaceres, CV_8U, 3);
imshow("(g) Laplacian of Gussian", glaplaceres);
imwrite("gLoG.jpg", glaplaceres);

Sobel(gimage, ggradx, CV_8U, 1, 0, 3);
Sobel(gimage, ggrady, CV_8U, 0, 1, 3);
imshow("Result of Sobel x kernelsize (g)", ggradx);
imshow("Result of Sobel y kernelsize (g)", ggrady);
imwrite("ggradx.jpg",ggradx);
imwrite("ggrady.jpg",ggrady);

ggradient = abs(ggradx) + abs(ggrady);
imshow("The gradient (g)", ggradient);
imwrite("ggradient.jpg", ggradient);

Canny(gimage, gcannyimage, 100, 230, 3);
imshow("canny result (g)", gcannyimage);
imwrite("gcannyimage.jpg", gcannyimage);

waitKey(0);
}

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;


int main() {
//-------------------------------------------------------------------------------
//part: 2-1:
//-------------------------------------------------------------------------------
  Mat fig2, image, resline;
  fig2 =  imread("Fig2.jpg");
  image = imread("Fig2.jpg");
  GaussianBlur(image, image, Size(9,9),2);
  Canny(image, resline, 100, 210, 3);
  imshow("canny implemented", resline);

  vector <Vec4i> lin;
  HoughLinesP(resline, lin, 1, CV_PI/180, 50, 120, 18);

  for(size_t i=0; i < lin.size(); i++){
    Vec4i l;
    l = lin[i];
    line(fig2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0), 7, CV_AA);
  }
  imshow("hougline result", fig2);
  imwrite("houghline.jpg",fig2);
//-------------------------------------------------------------------------------
//part: 2-1:
//-------------------------------------------------------------------------------
//*******************************Fig3:
  Mat fig3 = imread("Fig3.jpg");
  Mat im3 = imread("Fig3.jpg", 0), im3b;
  threshold( im3, im3b, 10,255,THRESH_BINARY );
  GaussianBlur(im3b, im3b, Size(9,9), 2);
  vector< Vec3f> cir3;
  HoughCircles(im3b, cir3, CV_HOUGH_GRADIENT, 1, 20, 150, 30, 0, 150);
  vector< Vec3f>::const_iterator itc3 = cir3.begin();
  while (itc3 != cir3.end()) {
    circle(fig3, Point((*itc3)[0], (*itc3)[1]), (*itc3)[2], Scalar(255,255,255), 2);
    ++itc3;
  }
  imshow("Detected Circles for Fig3", fig3);
  imwrite("houghcircle3.jpg",fig3);
//*******************************Fig4:
  Mat fig4 = imread("Fig4.jpg");
  Mat im4 = imread("Fig4.jpg", 0), im4b;
  GaussianBlur(im4, im4b, Size(9,9), 2);
  vector< Vec3f> cir4;
  HoughCircles(im4b, cir4, CV_HOUGH_GRADIENT, 1, 40, 150, 30, 0, 150);
  vector< Vec3f>::const_iterator itc4 = cir4.begin();
  while (itc4 != cir4.end()) {
    circle(fig4, Point((*itc4)[0], (*itc4)[1]), (*itc4)[2], Scalar(255,255,255), 2);
    ++itc4;
  }
  imshow("Detected Circles for Fig4", fig4);
  imwrite("houghcircle4.jpg",fig4);
//*******************************Fig5:
  Mat fig5 = imread("Fig5.jpg");
  Mat im5 = imread("Fig5.jpg", 0), im5b, im5l;
  GaussianBlur(im5, im5b, Size(9,9), 2.25);
  Laplacian(im5b, im5l, -1, 3);
  im5b = im5b - 1.1*im5l;
  vector< Vec3f> cir5;
  HoughCircles(im5b, cir5, CV_HOUGH_GRADIENT, 1, 20, 150, 12, 11, 25);
  vector< Vec3f>::const_iterator itc5 = cir5.begin();
  while (itc5 != cir5.end()) {
    circle(fig5, Point((*itc5)[0], (*itc5)[1]), (*itc5)[2], Scalar(255,255,255), 2);
    ++itc5;
  }
  imshow("Detected Circles for Fig5", fig5);
  imwrite("houghcircle5.jpg",fig5);

  waitKey(0);

}

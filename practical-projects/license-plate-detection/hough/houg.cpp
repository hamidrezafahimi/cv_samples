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
  fig2 =   imread("/home/hamidreza/Desktop/hough/ds/100.jpg");
  image =  imread("/home/hamidreza/Desktop/hough/ds/100.jpg");
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

  waitKey(0);

}

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(){

Mat in = imread("fig2.jpg",0);          // load grayscale
Mat dest;
Mat src_f;
in.convertTo(src_f,CV_32F);

int kernel_size = 30;
double sig = 1.5, th = 0, lm = 1, gm = 0.25, ps = 0;
cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
cv::filter2D(src_f, dest, CV_32F, kernel);

cerr << dest(Rect(30,30,10,10)) << endl; // peek into the data

Mat viz;
dest.convertTo(viz,CV_8U,1.0/255.0);     // move to proper[0..255] range to show it
imshow("k",kernel);
imshow("d",viz);
waitKey();

}

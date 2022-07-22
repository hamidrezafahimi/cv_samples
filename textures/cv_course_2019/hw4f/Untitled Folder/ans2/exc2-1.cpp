#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


int main(){

    Mat in = imread("fig1.jpg",0);

    cv::namedWindow("gabor kernel");
  	cv::namedWindow("gabor result - q: quit - s: save");

    int kernel_size = 20;
    cv::createTrackbar("kernel size", "gabor result - q: quit - s: save", &kernel_size, 200);
  	int sigma_bar = 20;
  	cv::createTrackbar("sigma", "gabor result - q: quit - s: save", &sigma_bar, 60);
  	int theta_bar = 90;
  	cv::createTrackbar("theta", "gabor result - q: quit - s: save", &theta_bar, 180);
  	int lambda_bar = 12;
  	cv::createTrackbar("lambda", "gabor result - q: quit - s: save", &lambda_bar, 25);
  	int gamma_bar = 0;
  	cv::createTrackbar("gamma", "gabor result - q: quit - s: save", &gamma_bar, 200);
  	int psi_bar = 180;
  	cv::createTrackbar("psi", "gabor result - q: quit - s: save", &psi_bar, 180);

  	for(;;)
  	{
      Mat dest;
      Mat src_f;
      Mat viz;
      in.convertTo(src_f,CV_32F);

  		cv::Mat kernel = cv::getGaborKernel(
  			cv::Size(kernel_size, kernel_size),
  			sigma_bar, theta_bar / 180. * M_PI,
  			lambda_bar, gamma_bar / 100., psi_bar / 180. * M_PI);

        cv::filter2D(src_f, dest, CV_32F, kernel);

  		cv::normalize(kernel, kernel, 1, 0, cv::NORM_MINMAX);
  		cv::Mat ker_im;
  		cv::resize(kernel, ker_im, cv::Size(512, 512));

      dest.convertTo(viz,CV_8U,1.0/255.0);
      cv::imshow("gabor kernel", ker_im);
  		cv::imshow("gabor result - q: quit - s: save", viz);
  		int key = cv::waitKey(20);

  		if(key == 'q')
  		{
  			break;
  		}
      else if (key == 's') {
        imwrite("out.jpg",viz);
        break;
      }
  	}

  	return 0;
}

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	// Read input image & show it:
	cv::Mat image= cv::imread("saltpepper2.jpg",0);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image",image);

	// Blur the image &  Display the blurred image:--------------------------------------------------
	cv::Mat result_gf3;
	cv::GaussianBlur(image,result_gf3,cv::Size(3,3),1.5);
	cv::namedWindow("Gaussian filtered Image gf3");
  cv::imwrite("gf3.jpg",result_gf3);
	cv::imshow("Gaussian filtered Image gf3",result_gf3);

  cv::Mat result_gf5;
  cv::GaussianBlur(image,result_gf5,cv::Size(5,5),1.5);
  cv::namedWindow("Gaussian filtered Image gf5");
  cv::imwrite("gf5.jpg",result_gf5);
  cv::imshow("Gaussian filtered Image gf5",result_gf5);

  cv::Mat result_gf7;
  cv::GaussianBlur(image,result_gf7,cv::Size(7,7),1.5);
  cv::namedWindow("Gaussian filtered Image gf7");
  cv::imwrite("gf7.jpg",result_gf7);
  cv::imshow("Gaussian filtered Image gf7",result_gf7);


	// Blur the image with a mean filter & Display the blurred image:----------------------------------
  cv::Mat result_mnf3;
	cv::blur(image,result_mnf3,cv::Size(3,3));
	cv::namedWindow("Mean filtered Image mnf3");
	cv::imwrite("mnf3.jpg",result_mnf3);
	cv::imshow("Mean filtered Image mnf3",result_mnf3);

	cv::Mat result_mnf5;
	cv::blur(image,result_mnf5,cv::Size(5,5));
	cv::namedWindow("Mean filtered Image mnf5");
	cv::imwrite("mnf5.jpg",result_mnf5);
	cv::imshow("Mean filtered Image mnf5",result_mnf5);

	cv::Mat result_mnf7;
	cv::blur(image,result_mnf7,cv::Size(7,7));
	cv::namedWindow("Mean filtered Image mnf7");
	cv::imwrite("mnf7.jpg",result_mnf7);
	cv::imshow("Mean filtered Image mnf7",result_mnf7);

	// Applying a median filter & Display the blurred image:-------------------------------------------
	cv::Mat result_mdf3;
	cv::medianBlur(image,result_mdf3,3);
	cv::namedWindow("Median filtered Image mdf3");
	cv::imwrite("mdf3.jpg",result_mdf3);
	cv::imshow("Median filtered Image mdf3",result_mdf3);

	cv::Mat result_mdf5;
	cv::medianBlur(image,result_mdf5,5);
	cv::namedWindow("Median filtered Image mdf5");
	cv::imwrite("mdf5.jpg",result_mdf5);
	cv::imshow("Median filtered Image mdf5",result_mdf5);

	cv::Mat result_mdf7;
	cv::medianBlur(image,result_mdf7,7);
	cv::namedWindow("Median filtered Image mdf7");
	cv::imwrite("mdf7.jpg",result_mdf7);
	cv::imshow("Median filtered Image mdf7",result_mdf7);

	cv::waitKey();
	return 0;
}


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
	using namespace std;

	cv::namedWindow("gabor");

	int sigma_bar = 40;
	cv::createTrackbar("sigma", "gabor", &sigma_bar, 1000, NULL, NULL);

	int theta_bar = 0;
	cv::createTrackbar("theta", "gabor", &theta_bar, 180, NULL, NULL);

	int lambda_bar = 11;
	cv::createTrackbar("lambda", "gabor", &lambda_bar, 100, NULL, NULL);

	int gamma_bar = 100;
	cv::createTrackbar("gamma", "gabor", &gamma_bar, 200, NULL, NULL);

	int psi_bar = 90;
	cv::createTrackbar("psi", "gabor", &psi_bar, 180, NULL, NULL);

	const int kernel_size = 128;

	for(;;)
	{
		cv::Mat gabor_mat = cv::getGaborKernel(
			cv::Size(kernel_size, kernel_size),
			sigma_bar / 10., theta_bar / 180. * M_PI,
			lambda_bar, gamma_bar / 100., psi_bar / 180. * M_PI);


		cv::normalize(gabor_mat, gabor_mat, 1, 0, cv::NORM_MINMAX);
		cv::Mat image;
		cv::resize(gabor_mat, image, cv::Size(512, 512));


		cv::imshow("gabor", image);
		int key = cv::waitKey(20);

		if(key == 'q')
		{
			break;
		}
	}

	return 0;
}

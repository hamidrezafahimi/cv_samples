
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

void saltpeper(cv::Mat &image, int ns, int np) {

	int i,j;
	for (int k=0; k<ns; k++) {
		i= rand()%image.cols;
		j= rand()%image.rows;
		if (image.channels() == 1) { // gray-level image
			image.at<uchar>(j,i)= 255;
		} else if (image.channels() == 3) { // color image
			image.at<cv::Vec3b>(j,i)[0]= 255;
			image.at<cv::Vec3b>(j,i)[1]= 255;
			image.at<cv::Vec3b>(j,i)[2]= 255;
		}
	}
	for (int k=0; k<np; k++) {
		i= rand()%image.cols;
		j= rand()%image.rows;
		if (image.channels() == 1) {
			image.at<uchar>(j,i)= 0;
		} else if (image.channels() == 3) {
			image.at<cv::Vec3b>(j,i)[0]= 0;
			image.at<cv::Vec3b>(j,i)[1]= 0;
			image.at<cv::Vec3b>(j,i)[2]= 0;
		}
	}
}

double uniform()
{
    return (rand()/(float)0x7fff)-0.5;
}

void uniformnoise(cv::Mat image, cv::Mat &uimage, float p) {
Mat uninoise(image.size(),image.type());
for(int y=0;y<image.rows;y++)
{
		for(int x=0;x<(3*image.cols);x++)
		{
			uninoise.at<uchar>(y,x) = (char)((uniform())*255);
		}
	}
	uimage = p*uninoise + image;
}


int main()
{
	srand(cv::getTickCount()); // init random number generator

	// gussian noise:-----------------------------------------------------------------------------------------
	cv::Mat image = cv::imread("fig1.jpg");
	cv::Mat gussianim1(image.size(),image.type());
	cv::Mat gussianim2(image.size(),image.type());
	cv::Mat gussianim3(image.size(),image.type());
	cv::Mat noise(image.size(),image.type());
	float m =(10);

	float sigma1 = (10);
	cv::randn(noise,m,sigma1);
	gussianim1 = image + noise;
	cv::imwrite("gussianim1.jpg",gussianim1);
	imshow("gussian noised image 1",gussianim1);

	float sigma2 = (25);
	cv::randn(noise,m,sigma2);
	gussianim2 = image + noise;
	cv::imwrite("gussianim2.jpg",gussianim2);
	imshow("gussian noised image 2",gussianim2);

	float sigma3 = (40);
	cv::randn(noise,m,sigma3);
	gussianim3 = image + noise;
	cv::imwrite("gussianim3.jpg",gussianim3);
	imshow("gussian noised image 3",gussianim3);

	// uniform noise:-----------------------------------------------------------------------------------------
	image = cv::imread("fig1.jpg");
	cv::Mat uniformim1(image.size(),image.type());
	uniformnoise(image,uniformim1,0.2);
	imwrite("uniformim1.jpg",uniformim1);
	imshow("uniform noised image 1",uniformim1);

	cv::Mat uniformim2(image.size(),image.type());
	uniformnoise(image,uniformim2,0.28);
	imwrite("uniformim2.jpg",uniformim2);
	imshow("uniform noised image 2",uniformim2);

	cv::Mat uniformim3(image.size(),image.type());
	uniformnoise(image,uniformim3,0.36);
	imwrite("uniformim3.jpg",uniformim3);
	imshow("uniform noised image 3",uniformim3);

	// salt & pepper noise:-----------------------------------------------------------------------------------
	image = cv::imread("fig1.jpg");
	cv::Mat saltpepperim1(image.size(),image.type());
	cv::Mat saltpepperim2(image.size(),image.type());
	cv::Mat saltpepperim3(image.size(),image.type());
	saltpepperim1 = image;
	saltpepperim2 = image;
  saltpepperim3 = image;

	saltpeper(saltpepperim1,1000,1000);
	cv::namedWindow("saltpepper Image 1");
  cv::imwrite("saltpepper1.jpg",saltpepperim1);
	cv::imshow("saltpepper Image 1",saltpepperim1);

	saltpeper(saltpepperim2,1750,1750);
	cv::namedWindow("saltpepper Image 2");
  cv::imwrite("saltpepper2.jpg",saltpepperim2);
	cv::imshow("saltpepper Image 2",saltpepperim2);

	saltpeper(saltpepperim3,2500,2500);
	cv::namedWindow("saltpepper Image 3");
	cv::imwrite("saltpepper3.jpg",saltpepperim3);
	cv::imshow("saltpepper Image 3",saltpepperim3);

	cv::waitKey();

	return 0;
}

// background subtraction using running average
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main()
{
	// Open the video file
	VideoCapture capture("vid.mp4");
	if (!capture.isOpened())
		return 1;

	bool stop(false);
	Mat frame, gray; //current video frame
	namedWindow("Extracted Frame");
	int id=0, delay = 1;

	Mat background, foreground_binary, foreground_mean, foreground_gussain,
	 		foreground_otsu, frbin, frgus, frmn, frot;

	while (!stop)
	{
		if (!capture.read(frame))
			break;
		cvtColor(frame, gray, CV_BGR2GRAY);
		imshow("Extracted Frame", gray);

		if (background.empty())
			gray.convertTo(background, CV_32F);

		Mat Temp;
		background.convertTo(Temp, CV_8U);

		//frame.convertTo(framechar, CV_8U);
		absdiff(Temp, gray, foreground_binary);
		foreground_gussain = foreground_binary.clone();
		foreground_mean = foreground_binary.clone();
		foreground_otsu = foreground_binary.clone();
		threshold(foreground_binary, foreground_binary, 10, 255, THRESH_BINARY);
		adaptiveThreshold(foreground_gussain,foreground_gussain, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 125, 1);
		adaptiveThreshold(foreground_mean,foreground_mean, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 125, 1);
		threshold(foreground_otsu,foreground_otsu, 25, 255, /*THRESH_BINARY+*/THRESH_OTSU);
		accumulateWeighted(gray, background, 0.3);

		imshow("foreground_binary", foreground_binary);
		imshow("foreground_gussain", foreground_gussain);
		imshow("foreground_mean", foreground_mean);
		imshow("foreground_otsu", foreground_otsu);

		if(id == 107) {foreground_binary.convertTo(frbin, CV_8U);
									 imwrite("bin.jpg", frbin);
									 foreground_gussain.convertTo(frgus, CV_8U);
							 		 imwrite("gus.jpg", frgus);
									 foreground_mean.convertTo(frmn, CV_8U);
							 		 imwrite("mn.jpg", frmn);
									 foreground_otsu.convertTo(frot, CV_8U);
									 imwrite("ot.jpg", frot);}

		if (waitKey(delay) >= 0)
			stop = true;

		++id;
	}
	capture.release();

}

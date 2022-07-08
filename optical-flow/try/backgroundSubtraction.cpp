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
	VideoCapture capture("HALL.avi");
	if (!capture.isOpened())
		return 1;

	bool stop(false);
	Mat gray; //current video frame
	namedWindow("Extracted Frame");
	int delay = 50;

	Mat background, foreground;

	while (!stop)
	{
		if (!capture.read(gray))
			break;
		resize(gray, gray, Size(gray.rows * 2, gray.cols * 2));
		imshow("Extracted Frame", gray);

		if (background.empty())
			gray.convertTo(background, CV_32F);

		Mat Temp;
		background.convertTo(Temp, CV_8U);

		absdiff(Temp, gray, foreground);
		threshold(foreground, foreground, 10, 255, THRESH_BINARY);
		accumulateWeighted(gray, background, 0.3);

		imshow("foreground", foreground);
		if (waitKey(delay) >= 0)
			stop = true;
	}
	capture.release();

}

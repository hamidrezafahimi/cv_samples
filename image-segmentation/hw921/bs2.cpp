//background subtraction using MOG
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/background_segm.hpp"
using namespace cv;
using namespace std;

int main()
{
	// Open the video file
	VideoCapture capture("vid.mp4");
	if (!capture.isOpened())
		return 1;

	bool stop(false);


	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	Mat img0, img, fgmask, fgimg;

	while (!stop)
	{
		capture >> img;

		if (img.empty())
			break;

		//resize(img0, img, Size(img0.rows*2, img0.cols*2), INTER_LINEAR);

		if (fgimg.empty())
			fgimg.create(img.size(), img.type());

		//update the model
		bg_model->apply(img, fgmask, -1);

		fgimg = Scalar::all(0);
		img.copyTo(fgimg, fgmask);

		Mat bgimg;
		bg_model->getBackgroundImage(bgimg);

		imshow("image", img);
		imshow("foreground mask", fgmask);
		imshow("foreground image", fgimg);
		if (!bgimg.empty())
			imshow("mean background image", bgimg);

		char k = (char)waitKey(70);
		if (k == 27) break;

	}
	capture.release();

}

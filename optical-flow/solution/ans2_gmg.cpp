//background subtraction using MOG
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/bgsegm.hpp"
using namespace cv;
using namespace std;

int main()
{
	vector<cv::String> fn;
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg", fn, true);
	glob("/home/hamidreza/cv/HW/hw7/JPEGS/boats/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/bottle/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/cyclists/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/surf/*.jpg", fn, true);
	vector<Mat> images;
	size_t count = fn.size(); //number of png files in images folder
	for (size_t i=0; i<count; i++){
			images.push_back(imread(fn[i]));}
	bool stop(false);

	Ptr<BackgroundSubtractor> bg_model = bgsegm::createBackgroundSubtractorGMG(20,0.5);
	Mat img0, img, fgmask, fgimg, bg;


	/// PART 2-1 ///
  int kernel_size = 3;
	// int kernel_size = 20;
	/// PART 2-1 ///

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size( 2*kernel_size + 1, 2*kernel_size+1 ));


	int j=0;
	while (!stop)
	{
		img0 = images[j];
		resize(img0, img, Size(img0.rows*4, img0.cols*2), INTER_LINEAR);
		if (fgimg.empty())
			fgimg.create(img.size(), img.type());
		bg_model->apply(img, fgmask, -1);
		fgimg = Scalar::all(0);
		img.copyTo(fgimg, fgmask);

		morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel);


    /// PART 2-3 ///
		absdiff(img, fgimg, bg);
		imshow("background: difference of image and foreground", bg);
		/// PART 2-3 ///

		imshow("image", img);
		imshow("foreground mask", fgmask);
		imshow("foreground image", fgimg);


		char k = (char)waitKey(200);
		if (k == 27) break;

		++j;

	}
	// capture.release();

}

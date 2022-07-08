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
	glob("/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/boats/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/bottle/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/cyclists/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/surf/*.jpg", fn, true);
	vector<Mat> images;
	size_t count = fn.size(); //number of png files in images folder
	for (size_t i=0; i<count; i++){
			images.push_back(imread(fn[i]));}
	bool stop(false);


	/// PART 2-2 ///
	// Ptr<BackgroundSubtractor> bg_model = bgsegm::createBackgroundSubtractorMOG(200, 2, 0.3, 0);
	// Ptr<BackgroundSubtractor> bg_model = bgsegm::createBackgroundSubtractorMOG(200, 3, 0.5, 0);
	Ptr<BackgroundSubtractor> bg_model = bgsegm::createBackgroundSubtractorMOG(200, 5, 0.7, 0);
	/// PART 2-2 ///

	Mat img0, img, fgmask, fgimg, bg;

	int take=25, j=0;
	while (!stop)
	{
		cout<<j<<endl;
		img0 = images[j];
		resize(img0, img, Size(img0.rows*4, img0.cols*2), INTER_LINEAR);
		if (fgimg.empty())
			fgimg.create(img.size(), img.type());
		bg_model->apply(img, fgmask, -1);
		fgimg = Scalar::all(0);
		img.copyTo(fgimg, fgmask);

		/// PART 2-3 ///
		absdiff(img, fgimg, bg);
		imshow("background: difference of image and foreground", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog_birds.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog_boats.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog_bottle.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog_cyclists.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog_surf.jpg", bg);
		/// PART 2-3 ///

		imshow("image", img);
		imshow("foreground mask", fgmask);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-2/fgm_mog_n5_bottle.jpg", fgmask);
		imshow("foreground image", fgimg);
		if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog_birds.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog_boats.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog_bottle.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog_cyclists.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog_surf.jpg", fgimg);



		char k = (char)waitKey(1);
		if (k == 27) break;

		++j;
	}
	// capture.release();

}

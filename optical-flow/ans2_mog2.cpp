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

///// read the collection
	vector<cv::String> fn;
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/boats/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/bottle/*.jpg", fn, true);
	// glob("/home/hamidreza/cv/HW/hw7/JPEGS/cyclists/*.jpg", fn, true);
	glob("/home/hamidreza/cv/HW/hw7/JPEGS/surf/*.jpg", fn, true);
	vector<Mat> images;
	size_t count = fn.size();
	for (size_t i=0; i<count; i++){
			images.push_back(imread(fn[i]));}


///// background subtraction
	bool stop(false);
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	Mat img0, img, fgmask, fgimg, bg;
	int take=26, j=0;
	while (!stop)
	{
		img0 = images[j];
		resize(img0, img, Size(img0.rows*4, img0.cols*2), INTER_LINEAR);
		if (fgimg.empty())
			fgimg.create(img.size(), img.type());
		bg_model->apply(img, fgmask, -1);
		fgimg = Scalar::all(0);
		img.copyTo(fgimg, fgmask);
		/// find mean background (only for MOG2)
		Mat bgimg;
		bg_model->getBackgroundImage(bgimg);


		///// PART 2-3 ///
		absdiff(img, fgimg, bg);
		imshow("background: difference of image and foreground", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog2_birds.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog2_boats.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog2_bottle.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog2_cyclists.jpg", bg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2-3/dif_mog2_surf.jpg", bg);
		///// PART 2-3 ///


///// output
		imshow("image", img);
		imshow("foreground mask for MOG2", fgmask);
		imshow("foreground image for MOG2", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog2_birds.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog2_boats.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog2_bottle.jpg", fgimg);
		// if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog2_cyclists.jpg", fgimg);
		if(j==take) imwrite("/home/hamidreza/cv/HW/hw7/output/2/fgi_mog2_surf.jpg", fgimg);



		if (!bgimg.empty())
			imshow("background: mean background image from MOG2", bgimg);


		char k = (char)waitKey(1);
		if (k == 27) break;
		++j;
	}
}

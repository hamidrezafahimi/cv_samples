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
	//namedWindow("Extracted Frame");
	int delay = 1;

	Mat background, framechar, foreground_binary, foreground_mean, foreground_gussain, foreground_otsu;
	//Mat frame;
	//capture.set(CV_CAP_PROP_POS_AVI_RATIO, 0.5);
	//capture.set(CV_CAP_PROP_POS_FRAMES, 110);
	capture.read(frame);
	//imshow("frame", frame);

	//waitKey(0);
	Mat frot;
/*
	VideoWriter write_otsu;
	write_otsu.open("otsu.AVI", -1, 25, frame.size(), 0);
*/
	int id=0;
	while (1)
	{
		if (!capture.read(frame))
			break;
		cvtColor(frame, gray, CV_BGR2GRAY);
		//imshow("Extracted Frame", gray);

		if (background.empty())
			gray.convertTo(background, CV_32F);

		Mat Temp;
		background.convertTo(Temp, CV_8U);

		frame.convertTo(framechar, CV_8U);
		absdiff(Temp, gray, foreground_binary);
		//threshold(foreground_otsu,foreground_otsu, 100, 255, THRESH_BINARY+THRESH_OTSU);
		threshold(foreground_binary,foreground_binary, 20, 255, THRESH_BINARY);
		accumulateWeighted(gray, background, 0.3);

		if(id == 933) {foreground_binary.convertTo(frot, CV_8U);
									 imwrite("ot.jpg", frot);}
		//write_otsu.write(foreground_otsu);
		//imshow("foreground_otsu", foreground_otsu);
/*
		if (waitKey(delay) >= 0)
			stop = true;
*/
		++id;
	}

	imshow("f", frot);

//**************************************************************************************************************************************************
/*
	Mat cont, dst, tmplate;
	Rect roi = selectROI(frot);
//	frot.convertTo(cont, CV_8UC1);

	GaussianBlur(frot, frot, Size(3,3), 1.5);

	//cvtColor(frot, cont, CV_BGR2GRAY);
	dst = frot.clone();
	vector <vector<Point> > contours, filteredcontours;
	vector <Vec4i> hierarchy;

	findContours(cont, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	for(int i=0; i<contours.size(); ++i){
		if(contourArea(contours[i]) > 5000)
		{
			vector <vector<Point> > filteredcontours;
			filteredcontours.push_back(contours[i]);
			drawContours(dst, filteredcontours, -1, Scalar(rand() & 255, rand() & 255,  rand() & 255));
		}
	}

	Rect r, f;

	int idx = 0, it=0;
	for (; idx >= 0; idx = hierarchy[idx][0]){
		r = boundingRect(Mat(contours[idx]));
		if((r.x>=roi.x && r.x<=(roi.x + roi.width)) &&
			 (r.y>=roi.y && r.y<=(roi.y + roi.height)) &&
			 (r.width <roi.width) && (r.height<roi.height)) {++it;
			f.x = r.x;
			f.y = r.y;
			f.width = r.width;
			f.height = r.height;
		}
	}
	cout<<it<<endl;
	tmplate = frot(Rect(f.x, f.y, f.width, f.height));
	//draw all contours with the same color

	imshow("result", dst);
	imshow("template", tmplate);


//*/

	waitKey(0);
	capture.release();

}

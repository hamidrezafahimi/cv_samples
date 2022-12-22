#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat src;

	src = imread("plate2.jpg", 0);
	threshold(src, src, 100, 255, THRESH_BINARY);
	//morphologyEx(src, src, MORPH_CLOSE, Mat(), Point(-1, -1), 2);

	imshow("Source", src);
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

		vector <vector<Point>> contours;
		vector <Vec4i> hierarchy;

	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//draw all contours with the same color
	drawContours(dst, contours, -1, Scalar(0, 0, 255));

	////draw each contour with a rondom color
	//int idx = 0;
	//for (; idx >= 0; idx = hierarchy[idx][0])
	//{
	//	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	//	if (isContourConvex(contours[idx]))
	//	drawContours(dst, contours, idx, color,0,8,hierarchy );
	//}
	//imshow("components", dst);

	//computing shape descriptors
	float radius;
	Point2f center;
	minEnclosingCircle(Mat(contours[1]), center, radius);
	circle(dst, Point(center), static_cast<int>(radius), Scalar(255,0,0), 2);

	RotatedRect rrect= fitEllipse(Mat(contours[2]));
	ellipse(dst,rrect,Scalar(255,0,0),2);


	// testing the approximate polygon
	vector<Point> poly;
	approxPolyDP(Mat(contours[3]), poly, 5, true);

	cout << "Polygon size: " << poly.size() << endl;

	// Iterate over each segment and draw it
	vector<Point>::const_iterator itp = poly.begin();
	while (itp != (poly.end() - 1)) {
		line(dst, *itp, *(itp + 1), Scalar(255,0,0), 2);
		++itp;
	}
	// last point linked to first point
	line(dst, *(poly.begin()), *(poly.end() - 1), Scalar(100,0,0), 2);

	// testing the convex hull
	vector<Point> hull;
	convexHull(Mat(contours[4]), hull);

	// Iterate over each segment and draw it
	vector<Point>::const_iterator it = hull.begin();
	while (it != (hull.end() - 1)) {
		line(dst, *it, *(it + 1), Scalar(100,0,0), 2);
		++it;
	}
	// last point linked to first point
	line(dst, *(hull.begin()), *(hull.end() - 1), Scalar(100,0,0), 2);

	// testing the moments

	// iterate over all contours
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()) {

		// compute all moments
		Moments mom = moments(Mat(*itc++));

		// draw mass center
		circle(dst,
			// position of mass center converted to integer
			Point(mom.m10 / mom.m00, mom.m01 / mom.m00),
			2, Scalar(150,0,0), 2); // draw black dot
	}

	namedWindow("Some Shape descriptors");
	imshow("Some Shape descriptors", dst);

	waitKey(0);
}

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

Mat frame;
Point pt(-1,-1);
bool newCoords = false;

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        // Store point coordinates
        pt.x = x;
        pt.y = y;
        newCoords = true;
    }
}

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("img", 1);

    // Set callback
    setMouseCallback("img", mouse_callback);

    for (;;)
    {
        cap >> frame; // get a new frame from camera

        // Show last point clicked, if valid
        if (pt.x != -1 && pt.y != -1)
        {
            circle(frame, pt, 3, Scalar(0, 0, 255));

            if (newCoords)
            {
                std::cout << "Clicked coordinates: " << pt << std::endl;
                newCoords = false;
            }
        }

        imshow("img", frame);

        // Exit if 'q' is pressed
        if ((waitKey(1) & 0xFF) == 'q') break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
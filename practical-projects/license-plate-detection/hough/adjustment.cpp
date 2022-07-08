



#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H1 = 95, low_H = 95, low_S = 81, low_V = 115;
int high_H1 = 113, high_H = 113, high_S = 255, high_V = 255;
RNG rng(12345);

enum State {
  StateDefault,
  StateIncLowH,
  StateIncHighH,
  StateDecLowV,
  StateSuccess,
  StateFail,
  StateEnd};

// changethresh:
//     low_H++;

double mindis_con(vector<Point> conto){
  double hom, mindis=1e9;
  for(int i =0; i<conto.size(); ++i){
    for(int j =0; j<conto.size(); ++j){
      if(i==j) continue;
      hom = ((conto[i].x - conto[j].x)*(conto[i].x - conto[j].x))+((conto[i].y - conto[j].y)*(conto[i].y - conto[j].y));
      hom = sqrt(hom);
      // cout<<"hom for mindis "<<hom<<endl;
      if(mindis>hom) mindis = hom;
    }
  }
  return mindis;
}

double maxdis_con(vector<Point> conto){
  double hom, maxdis=0;
  for(int i =0; i<conto.size(); ++i){
    for(int j =0; j<conto.size(); ++j){
      hom = ((conto[i].x - conto[j].x)*(conto[i].x - conto[j].x))+((conto[i].y - conto[j].y)*(conto[i].y - conto[j].y));
      hom = sqrt(hom);
      if(maxdis<hom) maxdis = hom;
    }
  }
  return maxdis;
}



static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}
int main(int argc, char* argv[])
{
    VideoCapture cap(0);
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    Mat frame, frame_HSV, frame_threshold;
    int chv = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    string imgNum = argv[1];
    frame = imread("/home/hamidreza/Desktop/hough/ds/" + imgNum + ".jpg");
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    // Convert from BGR to HSV colorspace

    Mat hsv[3], cannied;   //destination array
    Mat equalized_frame, equalized_frame_rgb;
    split(frame_HSV,hsv);

    // equalizeHist( hsv[0], hsv[0] );
    equalizeHist( hsv[2], hsv[2] );
    merge(hsv,3,equalized_frame);
    cvtColor(equalized_frame, equalized_frame_rgb, COLOR_HSV2BGR);


    State stateParam=StateDefault;
    while (true) {
        // Detect the object based on HSV Range Values
        inRange(equalized_frame, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);

        int erosion_size = 3;
        Mat element = getStructuringElement( MORPH_RECT,
                                            Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                            Point( erosion_size, erosion_size ) );

        /// Apply the erosion operation
        dilate( frame_threshold, frame_threshold, element );

        Canny(frame_threshold, cannied, 50, 210, 3);


        int erosion_size_C = 1;
        Mat element_C = getStructuringElement( MORPH_RECT,
                                            Size( 2*erosion_size_C + 1, 2*erosion_size_C+1 ),
                                            Point( erosion_size_C, erosion_size_C ) );
        dilate( cannied, cannied, element_C );



        findContours( cannied, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );

        Rect *plate = NULL;
        int offset;
        Mat drawing = frame.clone();
        Mat drawing1 = frame.clone();
        int cnt4=0, cnt3=0;
        for( int i = 0; i< contours.size(); i++ )
            {
              if( contourArea(contours[i]) < 100 || contourArea(contours[i]) > 5000 ) continue;

              Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
              drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );

              vector<Point> conto;
              approxPolyDP(Mat(contours[i]), conto, 13, true);
              std::cout << conto.size() << '\n';
              if(conto.size() == 4)
              {
                offset = 10;

                float minx=1e9, miny=1e9, maxx=0, maxy=0;
                for(int ii=0; ii<conto.size(); ii++){
                  if(minx>conto[ii].x) minx = conto[ii].x;
                  if(miny>conto[ii].y) miny = conto[ii].y;
                  if(maxx<conto[ii].x) maxx = conto[ii].x;
                  if(maxy<conto[ii].y) maxy = conto[ii].y;
                  }
                  // float ar = (maxy-miny) / (maxx-minx);

                float dx = abs(conto[2].x - conto[0].x);
                float dy = abs(conto[2].y - conto[0].y);
                float ar = dy / dx;

                cout<<"ar for conto four "<<ar<<endl;
                if (ar > 1.1 && ar < 3)
                {
                  ++cnt4;
                  // Rect plate((conto[0].x - offset), (conto[0].y - offset), 5*(conto[2].y - conto[0].y)+2*offset, (conto[2].y - conto[0].y)+2*offset);
                  plate = new Rect((minx - offset), (miny - offset), 5*(maxy - miny)+4*offset, (maxy - miny)+2*offset);
                  rectangle(drawing1,conto[0],conto[2],Scalar(0,0,255),3);
                  rectangle(drawing1, *plate, cv::Scalar(0, 255, 0),3);
                  break;
                }
              }
              if(conto.size()==3 && contours.size()<4){

                float ar;
                ar = maxdis_con(conto)/mindis_con(conto);
                cout<<"mindis"<<mindis_con(conto)<<endl;
                cout<<"maxdis"<<maxdis_con(conto)<<endl;
                cout<<"ar"<<ar<<endl;
                if(ar>1.3 && ar<3.7){
                  ++cnt3;
                  double minx=1e9, miny=1e9, maxx=0, maxy=0, flag=0;
                  for(int ii=0; ii<conto.size(); ii++){
                    // if(!(conto[ii].x>offset && conto[ii].y>offset)) continue;
                    if(minx>conto[ii].x) minx = conto[ii].x;
                    if(miny>conto[ii].y) miny = conto[ii].y;
                    if(maxx<conto[ii].x) maxx = conto[ii].x;
                    if(maxy<conto[ii].y) maxy = conto[ii].y;
                    // flag = 1;
                    }
                    cout<<"minx"<<minx<<endl;
                    cout<<"miny"<<miny<<endl;
                    cout<<"maxx"<<maxx<<endl;
                    cout<<"maxy"<<maxy<<endl;
                    offset = 50;

                // if(flag!=0) plate = new Rect((minx- 0.4*offset), (miny - offset), 5*abs(maxy - miny)+4*offset, abs(maxy - miny)+1.5*offset);
                // cout<<"kkk"<<(conto[0].x - offset)<<endl;
                // cout<<"mmm"<<(conto[0].y - offset)<<endl;
                // cout<<"fff"<<(5*abs(conto[2].y - conto[0].y)+2*offset)<<endl;
                // cout<<"sss"<<(abs(conto[2].y - conto[0].y)+2*offset)<<endl;
                plate = new Rect((conto[0].x - offset), conto[0].y - offset, 5*(conto[2].y - conto[0].y)+2*offset, (conto[2].y - conto[0].y)+2*offset);
                rectangle(drawing1,conto[0],conto[1],Scalar(0,0,255),3);
                rectangle(drawing1, *plate, cv::Scalar(0, 255, 0),3);
                break;
                }
              }
              // else if
            }
            //

            if(plate!=NULL){
              Mat cropped = frame(*plate);
              imshow("the plate", cropped);}




        // if(cnt4==0 && cnt3==0) {if(low_H >= high_H) {low_H = low_H1;
        //                                              if(!(high_H>=255)) ++high_H;
        //                                              else {high_H = high_H1;
        //                                                    chv++;}
        //                                               }
        //                         else if(chv==0) {++low_H;
        //                         cout<<"new lh "<<low_H<<endl;}
        //                         else { if(low_V ==0) break;
        //                                     else --low_V;
        //                                     cout<<"new lv "<<low_V<<endl;}
        //                         }
        // Show the frames
        // imshow(window_capture_name, frame);
        // imshow("equalized frame", equalized_frame_rgb);
        imshow(window_detection_name, frame_threshold);
        imshow("canny result", cannied);
        imshow("cont result", drawing);
        imshow("rect result", drawing1);
        char key = (char) waitKey(1);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }
    return 0;
}

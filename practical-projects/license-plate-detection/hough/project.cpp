#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 105, low_S = 81, low_V = 115;
int high_H = 116, high_S = 255, high_V = 255;
RNG rng(12345);


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


int main(int argc, char* argv[])
{

    Mat frame, frame_HSV, frame_threshold;
    string imgNum = argv[1];
    frame = imread("/home/hamidreza/Desktop/hough/ds/" + imgNum + ".jpg");


    // Convert from BGR to HSV colorspace
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);

    Mat hsv[3], cannied;   //destination array
    Mat equalized_frame, equalized_frame_rgb;
    split(frame_HSV,hsv);

        // equalizeHist( hsv[0], hsv[0] );
    equalizeHist( hsv[2], hsv[2] );
    merge(hsv,3,equalized_frame);
    cvtColor(equalized_frame, equalized_frame_rgb, COLOR_HSV2BGR);

        // Detect the object based on HSV Range Values
    inRange(equalized_frame, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);

    int d_size = 3;
    Mat element = getStructuringElement( MORPH_RECT,
                                        Size( 2*d_size + 1, 2*d_size+1 ),
                                        Point( d_size, d_size ) );

        /// Apply the erosion operation
    dilate( frame_threshold, frame_threshold, element );

    Canny(frame_threshold, cannied, 50, 210, 3);


    int d_size_C = 1;
    Mat element_C = getStructuringElement( MORPH_RECT,
                                        Size( 2*d_size_C + 1, 2*d_size_C+1 ),
                                        Point( d_size_C, d_size_C ) );
    dilate( cannied, cannied, element_C );



    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( cannied, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );

    Rect *plate = NULL;
    float offset = 10;
    Mat drawing_cont = frame.clone();
    Mat drawin_rect = frame.clone();
    for( int i = 0; i< contours.size(); i++ )
            {
              if( contourArea(contours[i]) < 100 || contourArea(contours[i]) > 5000 ) continue;

              Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
              drawContours( drawing_cont, contours, i, color, 2, 8, hierarchy, 0, Point() );

              vector<Point> conto;
              approxPolyDP(Mat(contours[i]), conto, 13, true);
              std::cout <<"conto size" <<conto.size() << '\n';
              if(conto.size() == 4)
              {
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

                cout<<ar<<endl;
                if (ar > 1.3 && ar < 3)
                {

                  // Rect plate((conto[0].x - offset), (conto[0].y - offset), 5*(conto[2].y - conto[0].y)+2*offset, (conto[2].y - conto[0].y)+2*offset);
                  plate = new Rect((minx - offset), (miny - offset), 5*abs(maxy - miny)+4*offset, abs(maxy - miny)+2*offset);
                  rectangle(drawin_rect,conto[0],conto[2],Scalar(0,0,255),3);
                  rectangle(drawin_rect, *plate, cv::Scalar(0, 255, 0),3);
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
                  int minx=1e9, miny=1e9, maxx=0, maxy=0, flag=0;
                  for(int ii=0; ii<conto.size(); ii++){
                    // if(!(conto[ii].x>offset && conto[ii].y>offset)) continue;
                    if(minx>conto[ii].x) minx = conto[ii].x;
                    if(miny>conto[ii].y) miny = conto[ii].y;
                    if(maxx<conto[ii].x) maxx = conto[ii].x;
                    if(maxy<conto[ii].y) maxy = conto[ii].y;
                    // flag = 1;
                    }
                offset = 50;
                // plate = new Rect((minx- 0.4*offset), (miny - offset), 5*abs(maxy - miny)+4*offset, abs(maxy - miny)+1.5*offset);
                plate = new Rect((conto[0].x - offset), (conto[0].y - offset), 5*(conto[2].y - conto[0].y)+2*offset, (conto[2].y - conto[0].y)+2*offset);
                rectangle(drawin_rect,conto[0],conto[2],Scalar(0,0,255),3);
                rectangle(drawin_rect, *plate, cv::Scalar(0, 255, 0),3);
                break;
                }
              }

          }

            if(plate!=NULL){
              Mat cropped = frame(*plate);
              imshow("the plate", cropped);}


        // Show the frames
        // imshow(window_capture_name, frame);
        // imshow("equalized frame", equalized_frame_rgb);
        // imshow(window_detection_name, frame_threshold);
        // imshow("canny result", cannied);
        // imshow("cont result", drawing_cont);
        imshow("rect result", drawin_rect);
        char key = (char) waitKey();
        // if (key == 'q' || key == 27)
        // {
        //     break;
        // }

    return 0;
}

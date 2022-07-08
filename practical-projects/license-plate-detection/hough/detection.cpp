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
const int low_H1 = 85, low_S1 = 81, low_V1 = 115;
const int high_H1 = 108, high_S1 = 255, high_V1 = 255;



RNG rng(12345);

enum State {
  StateDefault,
  StateIncLowH,
  StateIncHighH,
  StateDecLowV,
  StateSuccess,
  StateFail,
  StateEnd};

struct HsvRange {
  int lowH=0;
  int lowS=0;
  int lowV=0;
  int highH=0;
  int highS=0;
  int highV=0;

};

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


bool processImage(HsvRange hsvRange, Mat frame, Mat& drawingOut, Mat& croppedOut){
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;


  Mat frame_HSV, frame_threshold;
  int chv = 0;

  cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
  // Convert from BGR to HSV colorspace

  Mat hsv[3], cannied;   //destination array
  Mat equalized_frame, equalized_frame_rgb;
  split(frame_HSV,hsv);

  // equalizeHist( hsv[0], hsv[0] );
  equalizeHist( hsv[2], hsv[2] );
  merge(hsv,3,equalized_frame);
  cvtColor(equalized_frame, equalized_frame_rgb, COLOR_HSV2BGR);

  int low_H = hsvRange.lowH, low_S= hsvRange.lowS, low_V = hsvRange.lowV;
  int high_H = hsvRange.highH, high_S = hsvRange.highS, high_V = hsvRange.highV;


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
        std::cout << "conto size " <<conto.size() << '\n';
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
          offset = dy/4;

          cout<<"ar for conto four "<<ar<<endl;
          if (ar > 1.1 && ar < 3)
          {
            ++cnt4;
            // Rect plate((conto[0].x - offset), (conto[0].y - offset), 5*(conto[2].y - conto[0].y)+2*offset, (conto[2].y - conto[0].y)+2*offset);
            plate = new Rect((minx - offset), (miny - offset), 5*(maxy - miny)+4*offset, (maxy - miny)+2*offset);
            // rectangle(drawing1,conto[0],conto[2],Scalar(0,0,255),3);
            rectangle(drawing1, *plate, cv::Scalar(0, 255, 0),3);
            break;
          }
        }
        if(conto.size()==3 && contours.size()<4){

          float ar;
          ar = maxdis_con(conto)/mindis_con(conto);
          cout<<"mindis"<<mindis_con(conto)<<endl;
          cout<<"maxdis"<<maxdis_con(conto)<<endl;
          cout<<"ar for conto three"<<ar<<endl;
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
          // rectangle(drawing1,conto[0],conto[1],Scalar(0,0,255),3);
          rectangle(drawing1, *plate, cv::Scalar(0, 255, 0),3);
          break;
          }
        }
        // else if
      }
      //

      if(plate!=NULL){
        Mat cropped = frame(*plate);
        // cout<<"imshow cropped "<<__LINE__<<endl;
        imshow("the plate", cropped);
        delete plate;
        plate = NULL;
        // drawing1.copyTo(drawingOut);
        // cropped.copyTo(croppedOut);
        drawingOut = drawing1.clone();
        croppedOut = cropped.clone();
        }

        // drawingOut = new Mat;


        // cout<<"imshow cannied "<<__LINE__<<endl;
        // imshow("canny result", cannied);
        // cout<<"imshow cont "<<__LINE__<<endl;
        // imshow("cont result", drawing);

        // imshow("rect result", drawing1);

        char key = (char) waitKey(1);

        if(cnt4==0 && cnt3==0) return false;
        return true;

}


int main(int argc, char* argv[])
{

    Mat frame;

    string imgNum = argv[1];
    frame = imread("/home/hamidreza/Desktop/hough/ds/" + imgNum + ".jpg");

    State stateParam=StateDefault;

    HsvRange hsvRange;
    hsvRange.lowH=low_H1;
    hsvRange.lowS=low_S1;
    hsvRange.lowV=low_V1;
    hsvRange.highH=high_H1;
    hsvRange.highS=high_S1;
    hsvRange.highV=high_V1;

    bool run = true;
  Mat draw, crop;
  // int e_size = 1;
  // Mat element = getStructuringElement( MORPH_RECT,
  //                                     Size( 2*e_size + 1, 2*e_size+1 ),
  //                                     Point( e_size, e_size ) );
  //
  // vector<vector<Point> > contoursgp;
  // vector<Vec4i> hierarchygp;
  //
  // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  // vector<Point> conto;
    while (run) {


      // draw = new Mat;
      // crop = new Mat;


      Mat drawing, cropped, gray_crop, canniedGP;
      switch (stateParam) {
        case StateDefault:
          cout<<"def "<<endl;
          if(processImage(hsvRange, frame, draw, crop)) stateParam = StateSuccess;
          else stateParam = StateIncLowH;
          break;


        case StateIncLowH:
          if(hsvRange.lowH<hsvRange.highH) {
            ++hsvRange.lowH;
            cout<<"inc low h "<<hsvRange.lowH<<endl;
            if(processImage(hsvRange, frame, draw, crop)) stateParam = StateSuccess;
          }
          else {stateParam = StateIncHighH;
                hsvRange.lowH = low_H1;}
          break;


        case StateIncHighH:
        if(hsvRange.highH<175) {
          ++hsvRange.highH;
          cout<<"inc high h "<<hsvRange.highH<<endl;
          if(processImage(hsvRange, frame, draw, crop)) stateParam = StateSuccess;
        }
        else {stateParam = StateDecLowV;
              hsvRange.highH = high_H1;}
        break;


        case StateDecLowV:
        if(hsvRange.lowV>0) {
          --hsvRange.lowV;
          cout<<"inc low V "<<hsvRange.lowV<<endl;
          if(processImage(hsvRange, frame, draw, crop)) stateParam = StateSuccess;
        }
        else {stateParam = StateFail;
              hsvRange.lowV = low_V1;}
        break;


        case StateSuccess:
            cout<<"detection success"<<draw.size()<<endl;
            cvtColor(crop, gray_crop, CV_BGR2GRAY);
            equalizeHist(gray_crop, gray_crop);
            // threshold(crop,crop, 120, 255, THRESH_BINARY);
            adaptiveThreshold(gray_crop, gray_crop, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 53, 30);

            //
            // contoursgp.clear();
            // hierarchygp.clear();
            // Canny(gray_crop, canniedGP, 50, 210, 3);
            // findContours( canniedGP, contoursgp, hierarchygp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
            // for( int l = 0; l< contoursgp.size(); l++ )
            //     {
            //       if( contourArea(contoursgp[l]) < 30 /*|| contourArea(contoursgp[l]) > 5000*/ ) continue;
            //
            //       drawContours( crop, contoursgp, l, color, 2, 8, hierarchygp, 0, Point() );
            //
            //       conto.clear();
            //       approxPolyDP(Mat(contoursgp[l]), conto, 13, true);
            //     }

            imshow("rect result", draw);
            // imshow("contours result", crop);
            imshow("plate result", gray_crop);
            imwrite("pl.jpg", gray_crop);
            stateParam = StateEnd;
            break;


        case StateFail:
            cout<<"detection failed"<<endl;
            stateParam = StateEnd;
            break;


        case StateEnd:
            run = false;
            break;
      }




        char key = (char) waitKey(1);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }
    waitKey();
    return 0;
}

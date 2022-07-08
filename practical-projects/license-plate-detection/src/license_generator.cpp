#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/freetype.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include <stdlib.h> 

using namespace cv;
using namespace std;

const int dataset_num = 2000;
string alphabet[] = {"ب","ت","ج","چ","ح","خ","د","ذ","ر","ژ","س","ص","ض","ط","ظ","ع","غ","ق","ک","ح","خ","ل","م","ن","ی"};

static void onMouse(int event,int x,int y,int,void*)
{
    if (event == 1)
        cout << " " << x << " " << y << endl;
}

int generateRandomNum(int min, int max)
{
    double r = ((double) rand() / (RAND_MAX));

    return r * (max - min) + min; 
}

Mat randomPerspectiveTransform(Mat input, int cols, int rows, float ratio, Rect *box){

    int size_x = generateRandomNum( 0.3*cols, 0.7*cols );
    int size_y = ratio * size_x;
    Point center(generateRandomNum( size_x /2 + 1, cols - size_x/2), generateRandomNum(size_y/2 + 1, rows - size_y/2));


    // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4]; 
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
         
    // Lambda Matrix
    Mat lambda( 2, 4, CV_32FC1 );
    //Input and Output Image;
    Mat output;
     
    // Set the lambda matrix the same type and size as input
    lambda = Mat::zeros( rows, cols, input.type() );

    inputQuad[0] = Point2f( 0,0 );
    inputQuad[1] = Point2f( input.cols-1,0);
    inputQuad[2] = Point2f( input.cols-1,input.rows-1);
    inputQuad[3] = Point2f( 0, input.rows-1  );
 

    Point noise1(generateRandomNum(-cols/30 , cols/30), generateRandomNum(-cols/30 , cols/30));
    Point noise2(generateRandomNum(-cols/30 , cols/30), generateRandomNum(-cols/30 , cols/30));

    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = center + Point( -size_x/2, -size_y/2 ) + noise1;
    outputQuad[1] = center + Point( size_x/2, -size_y/2 ) + noise2;
    outputQuad[2] = center + Point( size_x/2, size_y/2 ) + noise2;
    outputQuad[3] = center + Point( -size_x/2, size_y/2 ) + noise1;

    std::vector<cv::Point2f> vec(outputQuad, outputQuad+4);
    *box = boundingRect(vec);
 
    // Get the Perspective Transform Matrix i.e. lambda 
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(input,output,lambda, Size(cols, rows));
 
    return output;
}

int main( int argc, char** argv )
{
    srand (time(NULL));

    Mat images[3];
    images[0] = imread("../license_template/pelak_white.jpeg", IMREAD_COLOR);   // Read the file
    images[1] = imread("../license_template/pelak_yellow.jpeg", IMREAD_COLOR); 
    images[2] = imread("../license_template/pelak_red.jpeg", IMREAD_COLOR); 

    resize(images[0], images[0], images[0].size() / 15);
    resize(images[1], images[1], images[1].size() / 15);
    resize(images[2], images[2], images[2].size() / 15);

    int normal_count = 0;
    int defaced_count = 0;

    for(int counter=1; counter<=dataset_num; counter++)
    {
        

        int random_number = rand()%10;
        int random_type = 0;

        int random_class = generateRandomNum(1,4);

        if (random_number == 1)
            random_type = 1;
        else if (random_number == 2)
            random_type = 2;


        Mat image = images[random_type].clone();
        cv::Scalar color;
        if (random_type == 2)
            color = cv::Scalar( 255, 255, 255 );
        else
            color = cv::Scalar( 0, 0, 1 );

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        cv::Ptr<cv::freetype::FreeType2> ft = freetype::createFreeType2();
        ft->loadFontData("../font/B Traffic Bold.ttf",0);

        ft->putText( image, to_string( ( rand()%89 + 11 ) ), cv::Point( 150, 0 ), 150, color, -1, cv::LINE_AA, false );
        
        string character = alphabet[rand()%25];
        ft->putText( image, character, cv::Point( 165 + 2*93, -10 ), 150, color, -1, cv::LINE_AA, false );
        ft->putText( image, to_string( ( rand()%899 + 101 ) ), cv::Point( 150 + 4*93, 0 ), 150, color, -1, cv::LINE_AA, false );
        
        ft->putText( image, to_string( ( rand()%89 + 11 ) ), cv::Point( 170 + 7*93, 50 ), 120, color, -1, cv::LINE_AA, false );

        cv::Mat noise(image.size(),image.type());
        float m = (generateRandomNum(-10,10),generateRandomNum(-10,10),generateRandomNum(-10,10));
        float sigma = (generateRandomNum(-10,10),generateRandomNum(-10,10),generateRandomNum(-10,10));
        cv::randn(noise, m, sigma); //mean and variance
        image += noise;

        blur(image, image, Size(generateRandomNum(2,5),generateRandomNum(2,5)));

        Mat streetImg = imread("../street/" + to_string( generateRandomNum(1,6594) ) + ".jpg" , IMREAD_COLOR); 

        // int height = generateRandomNum(250,480);
        // int width = generateRandomNum(300,640);
        int height = 240;
        int width = 320;
        int x = generateRandomNum(0,streetImg.cols - width);
        int y = generateRandomNum(0,streetImg.rows - height);
        Rect2d ROI(x,y,width,height);

        if (random_class == 2)
        {
            int left_x = generateRandomNum(150, 325);
            int right_x = left_x + generateRandomNum(200, 325);
            int top_y = 0;
            int bottom_y = image.rows;

            Scalar license_color(255,255,255);
            if (random_type == 1)
            {
                license_color = Scalar(0,255,255);
            } 
            else if (random_type == 2)
            {
                license_color = Scalar(0,0,255);
            }

            Scalar random_color(generateRandomNum(0,255),generateRandomNum(0,255),generateRandomNum(0,255));

            for (int i=left_x; i<right_x; i++)
            {
                for (int j=top_y; j<bottom_y; j++)
                {   
                    

                    if (generateRandomNum(1,4) == 1)
                    {
                        image.at<Vec3b>(j,i)[0] = license_color[0]; 
                        image.at<Vec3b>(j,i)[1] = license_color[1];
                        image.at<Vec3b>(j,i)[2] = license_color[2];
                    }
                    else
                    {
                        image.at<Vec3b>(j,i)[0] = random_color[0]; 
                        image.at<Vec3b>(j,i)[1] = random_color[1];
                        image.at<Vec3b>(j,i)[2] = random_color[2];
                    }

                }
            }
        }

        streetImg = streetImg(ROI);
        Rect boundingRect;

        Mat image1 = randomPerspectiveTransform(image, width, height, float(image.rows) / float(image.cols), &boundingRect);

        Mat dst = streetImg.clone();

        if (random_class == 1 || random_class == 2)
        {
            for (int i=0; i<width; i++)
            {
                for (int j=0; j<height; j++)
                {
                    if (image1.at<Vec3b>(j,i)[0] != 0 || 
                        image1.at<Vec3b>(j,i)[1] != 0 ||
                        image1.at<Vec3b>(j,i)[2] != 0)
                    {
                        dst.at<Vec3b>(j,i) = image1.at<Vec3b>(j,i);
                    }
                    else
                    {
                        dst.at<Vec3b>(j,i) = streetImg.at<Vec3b>(j,i);
                    }
                }
            }
        }

        blur(dst, dst, Size(generateRandomNum(2,4),generateRandomNum(2,4)));

        if (random_class == 1 || random_class == 2)
        {

            string line;
            string image_name;
            string label_name;
            float x = (boundingRect.x + boundingRect.width / 2) / float(width);
            float y = (boundingRect.y + boundingRect.height / 2) / float(height);
            float width1 = boundingRect.width / float(width);
            float height1 = boundingRect.height / float(height);

            if(random_class == 1)
            {
                normal_count++;
                line += "0 ";
                image_name = "../dataset/normal_" + to_string(normal_count) + ".jpg";
                label_name = "../dataset/normal_" + to_string(normal_count) + ".txt";
            }
            else if(random_class == 2)
            {
                defaced_count++;
                line += "1 ";
                image_name = "../dataset/defaced_" + to_string(defaced_count) + ".jpg";
                label_name = "../dataset/defaced_" + to_string(defaced_count) + ".txt";
            }

            line += to_string(x) + " ";
            line += to_string(y) + " ";
            line += to_string(width1) + " ";
            line += to_string(height1) + "\n";

            
            imwrite(image_name, dst);
            ofstream label_file;
            label_file.open (label_name);
            label_file << line;
            label_file.close();

            int percentage = float(float(counter) / dataset_num) * 100;
            cout << "Generating " << dataset_num << " images for dataset." << endl;
            cout << "Normal plates count: " << normal_count << endl;
            cout << "Defaced plates count: " << defaced_count << endl << endl;
            cout << "|";
            for(int per=0; per < 100; per++)
            {
                if (per < percentage)
                    cout << ">";
                else
                    cout << " ";
            }

            cout << "| " << percentage << "%" << " Generated ..." << endl;

            rectangle(dst, boundingRect, Scalar(0, 255, 0), 3);
            imshow( "Display window 2", dst );

            waitKey(); 
        }

    }

    cout << "Dataset Generated." << endl;

    return 0;
}
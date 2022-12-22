// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
//
// int main()
// {
//     Mat img = imread("/home/hamidreza/cv/HW/hw6/out3/aa.jpg");
//     Mat bins;
//     imshow("Original", img);
//
//     uchar N = 16;
//     img  /= N;
//     img  *= N;
//
//     imshow("Reduced", img);
//
//     Canny(img, bins, 100, 230, 3);
//
//     imshow("ced", bins);
//
//     waitKey();
//
//     return 0;
// }


#include <armadillo>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("/home/hamidreza/cv/HW/hw6/out3/aa.jpg");

    imshow("Original", img);

    // Cluster

    int K = 8;
    int n = img.rows * img.cols;
    Mat data = img.reshape(1, n);
    data.convertTo(data, CV_32F);

    vector<int> labels;
    Mat1f colors;
    kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

    for (int i = 0; i < n; ++i)
    {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }

    Mat reduced = data.reshape(3, img.rows);
    reduced.convertTo(reduced, CV_8U);


    imshow("Reduced", reduced);
    waitKey();

    return 0;
}

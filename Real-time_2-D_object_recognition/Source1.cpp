//#include <opencv2/opencv.hpp>
//#include "Header.h"
//
//using namespace cv;
//
//int main()
//{
//    // Read image
//    int n = 256, hist[256] = { 0 };
//    cv::Mat original_frame, blur_frame, gray_frame, threshold_frame, dilate_frame, erode_frame;
//    Mat image = imread("D:/study/computer_vision/project_3/Proj03Examples/0.png");
//    if (image.empty())
//    {
//        std::cerr << "Failed to read image file!" << std::endl;
//        return -1;
//    }
//
//    cv::GaussianBlur(original_frame, blur_frame, cv::Size(5, 5), 0);
//    cv::cvtColor(blur_frame, gray_frame, cv::COLOR_BGR2GRAY);
//    
//    for (int i = 0; i < gray_frame.rows; i++) {
//        uchar* temp_ptr = gray_frame.ptr<uchar>(i);
//        for (int j = 0; j < gray_frame.cols; j++) {
//            hist[int(temp_ptr[j])]++;
//        }
//    }
//    //for (int i=0;i<256;i++) {
//    //    std::cout << hist[i] << " ";
//    //}
//    //std::cout << "\n";
//    int valley_index = findlocal_min(hist);
//    //std::cout << valley_index<<" ";
//    gray_frame.copyTo(threshold_frame);
//    threshold(gray_frame, threshold_frame, valley_index);
//
//    // Find contours in the grayscale image
//    std::vector<std::vector<Point>> contours;
//    findContours(threshold_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    // Find the contour with the largest area
//    int maxAreaIndex = 0;
//    double maxArea = 0.0;
//    for (int i = 0; i < contours.size(); i++)
//    {
//        double area = contourArea(contours[i]);
//        if (area > maxArea)
//        {
//            maxArea = area;
//            maxAreaIndex = i;
//        }
//    }
//
//    // Find the minimum area bounding rectangle of the contour
//    RotatedRect boundingRect = minAreaRect(contours[maxAreaIndex]);
//
//    // Draw the bounding rectangle on the image
//    Point2f vertices[4];
//    boundingRect.points(vertices);
//    for (int i = 0; i < 4; i++)
//    {
//        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
//    }
//
//    // Display the image with the bounding rectangle
//    namedWindow("Bounding rectangle", WINDOW_NORMAL);
//    imshow("Bounding rectangle", image);
//    waitKey(0);
//    return 0;
//}

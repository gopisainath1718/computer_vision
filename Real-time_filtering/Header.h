#include<opencv2/opencv.hpp>
cv::Mat greyscale(cv::Mat& src, cv::Mat& dst);
cv::Mat blur5x5(cv::Mat& src, cv::Mat& dst);
cv::Mat sobelX3x3(cv::Mat& src, cv::Mat& dst);
cv::Mat sobelY3x3(cv::Mat& src, cv::Mat& dst);
cv::Mat magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
cv::Mat blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
cv::Mat cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);
cv::Mat mean3x3(cv::Mat& src, cv::Mat& dst);
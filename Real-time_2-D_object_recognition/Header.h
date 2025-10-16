#include<opencv2/opencv.hpp>
#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include <fstream>

cv::Mat threshold(cv::Mat copy);
cv::Mat color_segmentation(cv::Mat labels, int n);
std::vector<double> MOI(cv::Mat labels);
void drawLine(cv::Mat image, double x, double y, double alpha, cv::Scalar color);
void drawBoundingBox(cv::Mat original_frame, cv::Mat threshold_frame);
void writeToCSV(std::string filename, std::vector<std::string> classNamesDB, std::vector<std::vector<double>> featuresDB);
void loadFromCSV(std::string filename, std::vector<std::string>& classNamesDB, std::vector<std::vector<double>>& featuresDB);
std::string getClassName(char c);
double euclideanDistance(std::vector<double> features1, std::vector<double> features2);
std::string classifier(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature);
std::string classifierKNN(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature, int K);
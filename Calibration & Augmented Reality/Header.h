#pragma once
#include<opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>
#include<fstream>

void points_3d(bool patternfound, int& img_count, cv::Mat frame, std::vector<cv::Point2f>& corner_set, std::vector<std::vector<cv::Point2f> > &corner_list, std::vector<cv::Vec3f> &point_set, std::vector<std::vector<cv::Vec3f> > &point_list);
bool corners(cv::Mat frame, std::vector<cv::Point2f> &corner_set);
void calibration(cv::Mat frame, int img_count, std::vector<std::vector<cv::Vec3f> > point_list, std::vector<std::vector<cv::Point2f> > corner_list);
void writeCSV(std::string file_name, cv::Mat camera_matrix, cv::Vec<float, 5> dist_coef);
void readCSV(std::string file_name, cv::Mat camera_matrix, cv::Vec<float, 5> &dist_coef);

void  motion(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix, cv::Vec<float, 5>& dist_coef, float& m);
void letter_g(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix, cv::Vec<float, 5>& dist_coef);
void axes(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix, cv::Vec<float, 5>& dist_coef, std::vector<cv::Point2f>& corner_set);
void sphere(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix, cv::Vec<float, 5>& dist_coef, float& c);

void project_circle(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix);
void project_pyramid(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix);

void video(cv::Mat frame, cv::Mat img, std::vector<cv::Point2f>& corner_set);


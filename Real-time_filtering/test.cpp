//#include"Header.h"
//#include<iostream>
//#include<opencv2/opencv.hpp>
//
//
//int main() {
//
//		cv::Mat img = cv::imread("D:/pics/lenna.png");
//		cv::Mat dst, sx, sy;
//		img.copyTo(dst);
//		img.copyTo(sx);
//		img.copyTo(sy);
//		sx = sobelX3x3(img, sx);
//		sy = sobelY3x3(img, sy);
//		dst = magnitude(sx, sy, dst);
//		while (true)
//		{
//			cv::namedWindow("image1", cv::WINDOW_FREERATIO);
//			cv::namedWindow("image2", cv::WINDOW_FREERATIO);
//			cv::imshow("image1", img);
//			cv::imshow("image2", dst);
//
//			char ch = cv::waitKey(0);
//			if (ch == 'q') {
//				cv::destroyAllWindows();
//				break;
//			}
//		}
//	return 0;
//}
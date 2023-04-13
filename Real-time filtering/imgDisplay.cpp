//#include "Header.h"
//#include<iostream>
//#include<opencv2/opencv.hpp>
//
//int main() {
//	cv::Mat img = cv::imread("D:/pics/image.jpg");
//
//	while (true)
//	{
//		cv::namedWindow("image", cv::WINDOW_FREERATIO);
//		cv::imshow("image", img);
//		
//		char ch = cv::waitKey(0);
//		if (ch == 'q') {
//			cv::destroyAllWindows();
//			break;
//		}
//		if (ch == 'c') {
//			cv::imwrite("D:/pics/copied_image.png", img);
//			std::cout << "image copied..!";
//		}
//	}
//	return 0;
//}
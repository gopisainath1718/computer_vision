#include "Header.h"
#include<opencv2/opencv.hpp>
#include<iostream>

cv::Mat greyscale(cv::Mat& src, cv::Mat& dst) {

		for (int i = 0; i < src.rows; i++) {

			// create row pointers for accessing the rows.
			cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(i);
			cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

			for (int j = 0; j < src.cols; j++) {

				dst_ptr[j][0] = src_ptr[j][0];
				dst_ptr[j][1] = src_ptr[j][0];	 //copying single channel to all 3 channels
				dst_ptr[j][2] = src_ptr[j][0];
			}
		}
	return dst;
}

cv::Mat blur5x5(cv::Mat& src, cv::Mat& dst) {

	cv::Mat dst1;
	src.copyTo(dst1);

	for (int i = 2; i < src.rows - 3; i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* p2_ptr = src.ptr<cv::Vec3b>(i - 2);
		cv::Vec3b* p1_ptr = src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* n1_ptr = src.ptr<cv::Vec3b>(i + 1);
		cv::Vec3b* n2_ptr = src.ptr<cv::Vec3b>(i + 2);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {

				// applying convolution in horizontal direction
				dst_ptr[j][c] = 0.1 * p2_ptr[j][c] + 0.2 * p1_ptr[j][c] + 0.4 * ptr[j][c] +
								0.2 * n1_ptr[j][c] + 0.1 * n2_ptr[j][c];
			}
		}
	}
	
	for (int i = 0; i < dst.rows - 1; i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* pptr = dst.ptr<cv::Vec3b>(i);
		cv::Vec3b* dst1_ptr = dst1.ptr<cv::Vec3b>(i);

		for (int k = 2; k < dst.cols - 3; k++) {
			for (int a = 0; a < 3; a++) {

				// applying convolution in vertical direction
				dst1_ptr[k][a] = 0.1 * pptr[k-2][a] + 0.2 * pptr[k-1][a] + 0.4 * pptr[k][a] +
								 0.2 * pptr[k+1][a] + 0.1 * pptr[k+2][a];
			}
		}
	}
	return dst1;
}

cv::Mat sobelX3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat temp(src.rows, src.cols/2, CV_16SC3);

	for (int i = 1; i < src.rows - 2; i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* p1_ptr = src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* n1_ptr = src.ptr<cv::Vec3b>(i + 1);
		cv::Vec3b* temp_ptr = temp.ptr<cv::Vec3b>(i);

		for(int j =0;j<src.cols-1;j++) {
			for (int c=0;c<3;c++) {

				// applying convolution in horizontal direction
				temp_ptr[j][c] = 0.5 * p1_ptr[j][c] + ptr[j][c] + 0.5 * n1_ptr[j][c];
			}
		}
	}

	for (int i = 0 ;i<src.rows-1;i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* ptr = temp.ptr<cv::Vec3b>(i);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);
		for (int j=1;j<src.cols-2;j++) {
			for (int c = 0;c<3;c++) {

				// applying convolution in vertical direction
				dst_ptr[j][c] = 0.5 * ptr[j - 1][c] + -0.5 * ptr[j + 1][c];
			}
		}
	}
	return dst;
}
cv::Mat sobelY3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat temp(src.rows, src.cols / 2, CV_16SC3);

	for (int i = 0; i < src.rows - 1; i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* temp_ptr = temp.ptr<cv::Vec3b>(i);

		for (int j = 1; j < src.cols - 2; j++) {
			for (int c = 0; c < 3; c++) {

				// applying convolution in horizontal direction
				temp_ptr[j][c] = 0.5 * ptr[j - 1][c] + ptr[j][c] + 0.5 * ptr[j + 1][c];
			}
		}
	}

	for (int i = 1; i < src.rows - 2; i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* p1_ptr = temp.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* n1_ptr = temp.ptr<cv::Vec3b>(i + 1);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {

				// applying convolution in vertical direction
				dst_ptr[j][c] = 0.5 * p1_ptr[j][c] + -0.5 * n1_ptr[j][c];
			}
		}
	}
	return dst;
}

cv::Mat magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {

	for (int i = 0;i<dst.rows-1;i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* x_ptr = sx.ptr<cv::Vec3b>(i);
		cv::Vec3b* y_ptr = sy.ptr<cv::Vec3b>(i);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0;j<dst.cols-1;j++) {
			for (int c = 0;c<3;c++) {

				//calculating magnitude of each pixel and copying to dst frame
				dst_ptr[j][c] = std::sqrt(x_ptr[j][c]*x_ptr[j][c] + y_ptr[j][c]*y_ptr[j][c]);
			}
		}
	}
	return dst;
}

cv::Mat blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
	
	src = blur5x5(src, dst);	//blurring the image 
	int b = 255 / levels;
	for (int i = 0;i<src.rows-1;i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* src_ptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0;j<src.cols-1;j++) {
			for (int c = 0;c<3;c++) {

				//quantizing the pixels
				dst_ptr[j][c] /= b;
				dst_ptr[j][c] *= b;
			}
		}
	}
	return dst;
}

cv::Mat cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {

	cv::Mat dst1;
	dst.copyTo(dst1);
	src = blurQuantize(src, dst1, levels);  // quantizing the image
	cv::Mat sx = sobelX3x3(src, dst);
	cv::Mat sy = sobelY3x3(src, dst);
	cv::Mat mg = magnitude(sx, sy, dst);

	for (int i = 0;i<dst.rows-1;i++) {

		// create row pointers for accessing the rows
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);
		cv::Vec3b* mg_ptr = mg.ptr<cv::Vec3b>(i);
		cv::Vec3b* src_ptr = src.ptr<cv::Vec3b>(i);

		for (int j= 2;j<dst.cols-3;j++) {
			if (mg_ptr[j][0] > 255 - magThreshold || mg_ptr[j][1] > 255 - magThreshold || mg_ptr[j][2] > 255 - magThreshold) {
				dst_ptr[j][0] = 0;
				dst_ptr[j - 1][0] = 0;
				dst_ptr[j - 2][0] = 0;
				dst_ptr[j + 1][0] = 0;
				dst_ptr[j + 2][0] = 0;
				dst_ptr[j][1] = 0;				
				dst_ptr[j - 1][1] = 0;			// converting [j-1],[j-2],[j],[j+1],[j+2] pixels to black					
				dst_ptr[j - 2][1] = 0;			//				for getting more cartoon effect.	
				dst_ptr[j + 1][1] = 0;			
				dst_ptr[j + 2][1] = 0;		
				dst_ptr[j][2] = 0;
				dst_ptr[j - 1][2] = 0;
				dst_ptr[j - 2][2] = 0;
				dst_ptr[j + 1][2] = 0;
				dst_ptr[j + 2][2] = 0;
			}
			else {
				dst_ptr[j][0] = src_ptr[j][0];
				dst_ptr[j][1] = src_ptr[j][1];		//copying the quantized pixels
				dst_ptr[j][2] = src_ptr[j][2];
			}
		}
	}
	return dst;
}

cv::Mat mean3x3(cv::Mat& src, cv::Mat& dst) {

	for (int i = 1; i < src.rows - 2; i++) {

		// create row pointers for accessing the rows.
		cv::Vec3b* p1_ptr = src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* n1_ptr = src.ptr<cv::Vec3b>(i + 1);
		cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 1; j < src.cols - 2; j++) {														  // applying a 3X3 filter on to pixels
			for (int c = 0; c < 3; c++) {																	
				dst_ptr[j][c] = (1.0/9)*p1_ptr[j-1][c] + (1.0/9)*p1_ptr[j][c] + (1.0/9)*p1_ptr[j+1][c] +  //  [0.3][0.3][0.3]	 [i-1,j-1] [i-1,j] [i-1, j+1]
								(1.0/9)*ptr[j-1][c]    + (1.0/9)*ptr[j][c]    + (1.0/9)*ptr[j+1][c] +	  //  [0.3][0.3][0.3] -> [i, j-1]  [i,j]   [i,j+1]
								(1.0/9)*n1_ptr[j-1][c] + (1.0/9)*n1_ptr[j][c] + (1.0/9)*n1_ptr[j+1][c];	  //  [0.3][0.3][0.3]	 [i+1,j-1] [i+1,j] [i+1,j+1]
			}
		}
	}
	return dst;
}
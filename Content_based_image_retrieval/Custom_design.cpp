#include<iostream>
#include<opencv2/opencv.hpp>
#include "csv_util.h"
#include<vector>
#include<algorithm>

int main(int argc, char*argv[]) {

	//reading target images and extracting freature sets
	cv::Mat target_image1 = cv::imread(argv[1]);
	cv::Mat target_image2 = cv::imread(argv[2]);
	std::vector<float> feature_set = custom_design_feature(argv[1], argv[2]);
	
	//reading the feature vectors file
	char filename[256];
	std::vector<char*> image_paths;
	std::vector<std::vector<float>> data;
	strcpy(filename, argv[3]);
	read_image_data_csv(filename, image_paths, data, 0);

	//computing the distance metric and storing in a vector
    //using the maximum intersetion as the distance metrics 
	std::vector<std::vector<std::string>> matched_images;
		for (int i = 0; i < data.size(); i++) {
			int temp = 0, temp1 = 0, temp2 = 0;
			for (int j = 0; j <258; j++) {
				temp1 = temp1 + std::min(feature_set[j], data[i][j]);
			}
			for (int j = 259;j< data[i].size();j++) {
				temp2 = temp2 + std::min(feature_set[j], data[i][j]);
			}
			temp = (3*temp1+ 7*temp2) / 10;
			matched_images.push_back({ std::to_string(temp), image_paths[i] });
		}
	
	//sorting the vector from most similar image to least similar image
	std::sort(matched_images.begin(), matched_images.end(),
		[](std::vector<std::string>& a, std::vector<std::string>& b)
		{
			return stoi(a[0]) > stoi(b[0]);
		});

	//reading the top 10 images
	cv::Mat img1 = cv::imread(matched_images[1][1]);
	cv::Mat img2 = cv::imread(matched_images[2][1]);
	cv::Mat img3 = cv::imread(matched_images[3][1]);
	cv::Mat img4 = cv::imread(matched_images[4][1]);
	cv::Mat img5 = cv::imread(matched_images[5][1]);
	cv::Mat img6 = cv::imread(matched_images[6][1]);
	cv::Mat img7 = cv::imread(matched_images[7][1]);
	cv::Mat img8 = cv::imread(matched_images[8][1]);
	cv::Mat img9 = cv::imread(matched_images[9][1]);
	cv::Mat img10 = cv::imread(matched_images[10][1]);

	//displaying the top 10 images
	while (1) {
		cv::imshow("target_image1", target_image1);
		cv::imshow("target_image2", target_image2);
		cv::imshow("image1", img1);
		cv::imshow("image2", img2);
		cv::imshow("image3", img3);
		cv::imshow("image4", img4);
		cv::imshow("image5", img5);
		cv::imshow("image6", img6);
		cv::imshow("image7", img7);
		cv::imshow("image8", img8);
		cv::imshow("image9", img9);
		cv::imshow("image10", img10);

		if (cv::waitKey() == 'q') {
			cv::destroyAllWindows();
			break;
		}

	}
	return 0;
}
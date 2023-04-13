#include<iostream>
#include<opencv2/opencv.hpp>
#include "csv_util.h"
#include<vector>
#include<algorithm>

int main(int argc,char* argv[]) {

	//reading target image and extracting freature set
	cv::Mat target_image = cv::imread(argv[1]);
	std::vector<float> feature_set = baseline_feature(argv[1]);

	//reading the feature vectors file
	char filename[256];
	std::vector<char*> image_paths;
	std::vector<std::vector<float>> data;
	strcpy(filename, argv[2]);
	read_image_data_csv(filename,image_paths, data, 0);
	
	//computing the distance metric and storing in a vector
	//using the euclidean distance as the distance metrics 
	std::vector<std::vector<std::string>> matched_images;
	for (int i = 0; i < data.size(); i++) {
		int temp = 0;
		for (int j = 0; j < data[i].size(); j++) {

			int diff = feature_set[j] - data[i][j];
			temp += diff * diff;
		}
		matched_images.push_back({ std::to_string(temp), image_paths[i] });
	}

	//sorting the vector from most similar image to least similar image
	std::sort(matched_images.begin(),matched_images.end(),
		[](std::vector<std::string>& a, std::vector<std::string>& b)
		{
			return stoi(a[0]) < stoi(b[0]);
		});

	//printing the sorted image paths
	//for (int i = 0;i<1107;i++)
	//std::cout << matched_images[i][1]<<"\n";

	//reading the top 3 images
	cv::Mat img1 = cv::imread(matched_images[1][1]);
	cv::Mat img2 = cv::imread(matched_images[2][1]);
	cv::Mat img3 = cv::imread(matched_images[3][1]);

	//displaying the top 3 images
	while (1) {
		cv::imshow("target_image", target_image);
		cv::imshow("image1", img1);
		cv::imshow("image2", img2);
		cv::imshow("image3", img3);

		if (cv::waitKey() == 'q') {
			cv::destroyAllWindows();
			break;
		}

	}

	return 0;
}
#include<iostream>
#include<opencv2/opencv.hpp>
#include "csv_util.h"
#include<vector>
#include<algorithm>

int main(int arc, char* argv[]) {
	
	//reading target image and extracting freature set
    cv::Mat target_image = cv::imread(argv[1]);
    std::vector<float> feature_set = TC_feature(argv[1]);
    
	//reading the feature vectors file
    char filename[256];
    std::vector<char*> image_paths;
    std::vector<std::vector<float>> data;
    strcpy(filename, argv[2]);
    read_image_data_csv(filename, image_paths, data, 0);
    
	//computing the distance metric and storing in a vector
	//using maximum intersection as the distance metrics
    std::vector<std::vector<std::string>> matched_images;
	for (int i = 0; i < data.size(); i++) {
		int temp1 = 0, temp2 = 0, temp3 = 0, temp4 =0, temp5 =0;
		for (int j = 0; j <258; j++) {
			temp1 = temp1 + std::min(feature_set[j], data[i][j]);
		}
		for (int j = 259;j< 515;j++) {
			temp2 = temp2 + std::min(feature_set[j], data[i][j]);
		}
		for (int j = 516; j < 772; j++) {
			temp3 = temp3 + std::min(feature_set[j], data[i][j]);
		}
		for (int j = 773; j < 1029; j++) {
			temp4 = temp4 + std::min(feature_set[j], data[i][j]);
		}
		for (int j = 1030; j < data[i].size(); j++) {
			temp5 = temp5 + std::min(feature_set[j], data[i][j]);
		}
		int temp6 = std::max({ temp1, temp2, temp3, temp4, temp5 });
		float temp = (3*temp1+ 7*temp6)/10.0;
		matched_images.push_back({ std::to_string(temp), image_paths[i] });
	}

	//sorting the vector from most similar image to least similar image
	std::sort(matched_images.begin(), matched_images.end(),
		[](std::vector<std::string>& a, std::vector<std::string>& b)
		{
			return stoi(a[0]) > stoi(b[0]);
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
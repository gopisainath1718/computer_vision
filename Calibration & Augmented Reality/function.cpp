#include "Header.h"

void writeCSV(std::string file_name, cv::Mat camera_matrix, cv::Vec<float, 5> dist_coef) {
	std::ofstream file;
	file.open(file_name, std::ofstream::trunc);
	file << "camera_matrix, ";
	for (int i = 0; i < camera_matrix.rows; i++) {
		for (int j = 0; j < camera_matrix.cols; j++) {
			if (i == camera_matrix.rows - 1 && j == camera_matrix.cols - 1)
				file << camera_matrix.at<double>(i, j);
			else
			file << camera_matrix.at<double>(i, j) << ",";
		}
	}
	file<< "\ndist_coef, ";
	for(int i = 0; i<5; i++)
		if(i == 4)
			file << dist_coef[i];
		else
		file << dist_coef[i] << ",";
}

void readCSV(std::string file_name, cv::Mat camera_matrix, cv::Vec<float, 5> &dist_coef) {
	std::ifstream file(file_name);
	if (file.is_open()) {
		std::string line, word;
		std::vector<double>words;
		int pos = 0;
		std::string token;
		
		while (getline(file, line)) {
			std::stringstream str(line);
			while (getline(str, word, ',')) {
				if (word == "camera_matrix" || word == "dist_coef")
					continue;
				else
					words.push_back(std::stod(word));
			}
		}
		for (int i = 0; i < camera_matrix.rows; i++) {
			for (int j = 0; j < camera_matrix.cols; j++) {
				camera_matrix.at<double>(i, j) = words[pos];
				pos++;
			}
		}
		pos = 9;
		for (int i = 0; i < 5; i++) {
			dist_coef[i] = words[pos];
			pos++;
		}
	}
}

bool corners(cv::Mat frame, std::vector<cv::Point2f> &corner_set) {
	cv::Mat gray;
	cv::Size patternSize = cv::Size(9, 6);
	cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

	bool patternfound = cv::findChessboardCorners(gray, patternSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH);

	//for (int i = 0; i < corner_set.size(); i++) {
	//std::cout << "co-ordinates of first corner" << corner_set[0].x << ", " << corner_set[0].y << "\t";
	//    //std::cout << point_set[i] << "\n";
	//}
	//std::cout << corner_set.size()<<"\n";
	if (patternfound)
		cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

	cv::drawChessboardCorners(frame, patternSize, corner_set, patternfound);
	return patternfound;
}

void points_3d(bool patternfound, int& img_count, cv::Mat frame, std::vector<cv::Point2f>& corner_set, std::vector<std::vector<cv::Point2f> >& corner_list, std::vector<cv::Vec3f>& point_set, std::vector<std::vector<cv::Vec3f> >& point_list) {

	if (patternfound) {
		img_count++;
		std::string name = "image_" + std::to_string(img_count) + ".jpg";
		cv::imwrite(name, frame);
		corner_list.push_back(corner_set);
		for (int y = 0; y > -6; y--) {
			for (int x = 0; x < 9; x++) {
				//cv::Vec3f point = (x, y, 0);
				//std::cout << x << "," << y<<"\n";
				point_set.push_back(cv::Vec3f(x, y, 0));
			}
		}
		//for (int i = 0; i < corner_set.size(); i++) {
		//    std::cout << "co-ordinates of first corner" << corner_set[i].x << ", " << corner_set[i].y << "\t";
		//    std::cout << point_set[i] << "\n";
		//}
		std::cout << point_set.size() << "\n";
		point_list.push_back(point_set);
		//std::cout << "corner_list.size() " << corner_list.size() << "\n";
		//std::cout << "point_list.size() " << point_list.size() << "\n";
	}
	else
	{
		std::cout << "\n54 points not found.\n";
	}
}

void calibration(cv::Mat frame, int img_count, std::vector<std::vector<cv::Vec3f> > point_list, std::vector<std::vector<cv::Point2f> > corner_list) {

	cv::Size frame_size(640, 480);
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, frame.cols / 2, 0, 1, frame.rows / 2, 0, 0, 1);
	//std::cout << "C = " << std::endl << " " << camera_matrix << std::endl << std::endl;
	std::vector<float> /*dist_coef*/ sd_intrinsic, sd_extrinsic, per_view_errors;
	std::vector<cv::Mat> rvecs, tvecs;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON);
	int flags = cv::CALIB_FIX_ASPECT_RATIO;
	cv::Vec<float, 5> dist_coef(0, 0, 0, 0, 0);

	if (img_count > 5) {
		float error = cv::calibrateCamera(point_list, corner_list, frame_size, camera_matrix, dist_coef, rvecs, tvecs, flags);

		std::cout << "error= " << error << "\n camera_matrix= \n" << camera_matrix << "\n coef=\n" << dist_coef << "\n";

		writeCSV("parameters.csv", camera_matrix, dist_coef);
	}
	else
		/*std::cout << "Requires minimum of 5 images, current number of images : " << img_count<<"\n"*/;
}


void axes(cv::Mat frame, cv::Mat rvecs, cv::Mat tvecs, cv::Mat camera_matrix, cv::Vec<float, 5> &dist_coef, std::vector<cv::Point2f> &corner_set) {

	std::vector<cv::Vec3f> axes;
	axes.push_back(cv::Vec3f(0, 0, 0));
	axes.push_back(cv::Vec3f(2, 0, 0));
	axes.push_back(cv::Vec3f(0, 2, 0));      //points for 3-d axes
	axes.push_back(cv::Vec3f(0, 0, 2));

	std::vector<cv::Point2f> corner_set_1;

	cv::projectPoints(axes, rvecs, tvecs, camera_matrix, dist_coef, corner_set_1);

    cv::arrowedLine(frame, corner_set[0], corner_set_1[1], cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::arrowedLine(frame, corner_set[0], corner_set_1[2], cv::Scalar(0, 255, 0), 2, 8, 0);
    cv::arrowedLine(frame, corner_set[0], corner_set_1[3], cv::Scalar(0, 0, 255), 2, 8, 0);
}



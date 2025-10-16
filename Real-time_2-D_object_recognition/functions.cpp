#include"Header.h"

cv::Mat threshold(cv::Mat copy) {

    int sum = 0;
    int n = 256;
    int hist[256] = { 0 };

    cv::Mat final;
    copy.copyTo(final);

    for (int i = 0; i < copy.rows; i++) {
        uchar* temp_ptr = copy.ptr<uchar>(i);
        for (int j = 0; j < copy.cols; j++) {
            hist[int(temp_ptr[j])]++;
        }
    }

    int minimum = 0;
    int middle_index = n / 2;
    int max_index_1 = middle_index, max_index_2 = middle_index;
    int min_index_1 = middle_index, min_index_2 = middle_index;
    int max_1 = hist[middle_index], max_2 = hist[middle_index];
    int min_1 = hist[middle_index], min_2 = hist[middle_index];

    for (int i = middle_index - 1; i >= 0; i--) {
        if (hist[i] >= max_1) {
            max_1 = hist[i];
            max_index_1 = i;
        }
        else if (hist[i] < min_1) {
            min_1 = hist[i];
            min_index_1 = i;
        }
        else {
            break; // exit loop if no more local minimum possible
        }
    }
    for (int i = middle_index + 1; i < n; i++) {
        if (hist[i] >= max_2) {
            max_2 = hist[i];
            max_index_2 = i;
        }
        else if (hist[i] < min_2) {
            min_2 = hist[i];
            min_index_2 = i;
        }
        else {
            break; // exit loop if no more local minimum possible
        }
    }
    if (min_1 < min_2) {
        minimum = min_index_1;
    }
    else {
        minimum = min_index_2;
    }

    for (int i = 0; i < copy.rows; i++) {
        uchar* final_ptr = final.ptr<uchar>(i);
        for (int j = 0; j < copy.cols; j++) {
            if ((int)final_ptr[j] > minimum) {
                final_ptr[j] = 0;
            }
            else final_ptr[j] = 255;
        }
    }
    return final;
}

cv::Mat color_segmentation(cv::Mat labels, int n) {

    std::vector<cv::Vec3b> colors(4);
    colors[0] = cv::Vec3b(255, 0, 0);
    colors[1] = cv::Vec3b(0, 0, 255);
    colors[2] = cv::Vec3b(0, 255, 0);
    colors[3] = cv::Vec3b(255, 0, 0);

    cv::Mat out = cv::Mat::zeros(labels.size(), CV_8UC3);
    for (int k = 1; k < n; k++) {
        cv::Mat region_mask = (labels == k);
        for (int i = 0; i < labels.rows; i++) {
            uchar* region_ptr = region_mask.ptr<uchar>(i);
            cv::Vec3b* out_ptr = out.ptr<cv::Vec3b>(i);
            for (int j = 0; j < labels.cols; j++) {
                if (region_ptr[j] != 0)
                    out_ptr[j] = colors[k];
            }
        }
    }

    return out;
}

std::vector<double> MOI(cv::Mat labels) {

    cv::Mat region_mask = (labels == 1);

    cv::imshow("mask", region_mask);

    //Compute moments of the region
    cv::Moments moments = cv::moments(region_mask, true);

    std::vector<double> huMoments;

    HuMoments(moments, huMoments);

    // Log scale hu moments 
    for (int i = 0; i < 7; i++) {
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
    }

    float centroid_x = moments.m10 / moments.m00;
    float centroid_y = moments.m01 / moments.m00;
    //// Compute axis of least central moment
    float alpha = 0.5 * atan2(2 * moments.nu11, moments.nu20 - moments.nu02);
    //std::cout << alpha << " ";

    return huMoments;
}

void drawLine(cv::Mat image, double x, double y, double alpha, cv::Scalar color) {
    double length = 100.0;
    double edge1 = length * sin(alpha);
    double edge2 = sqrt(length * length - edge1 * edge1);
    double xPrime = x + edge2, yPrime = y + edge1;

    arrowedLine(image, cv::Point(x, y), cv::Point(xPrime, yPrime), color, 3);
}

void drawBoundingBox(cv::Mat original_frame, cv::Mat threshold_frame) {
    std::vector<std::vector<cv::Point>> contours;
    findContours(threshold_frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Find the contour with the largest area
    int maxAreaIndex = 0;
    double maxArea = 0.0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > maxArea)
        {
            maxArea = area;
            maxAreaIndex = i;
        }
    }
    // Find the minimum area bounding rectangle of the contour
    cv::RotatedRect boundingRect = minAreaRect(contours[maxAreaIndex]);
    // Draw the bounding rectangle on the image
    cv::Point2f vertices[4];
    boundingRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(original_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
}

void writeToCSV(std::string filename, std::vector<std::string> classNamesDB, std::vector<std::vector<double>> featuresDB) {
    // create an output filestream object
    std::ofstream csvFile;
    csvFile.open(filename, std::ofstream::trunc);

    // send data to the stream
    for (int i = 0; i < classNamesDB.size(); i++) {
        // add class name
        csvFile << classNamesDB[i] << ",";
        // add features
        for (int j = 0; j < featuresDB[i].size(); j++) {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) {
                csvFile << ","; // no comma at the end of line
            }
        }
        csvFile << "\n";
    }
}

void loadFromCSV(std::string filename, std::vector<std::string>& classNamesDB, std::vector<std::vector<double>>& featuresDB) {
    // create an input filestream object
    std::ifstream csvFile(filename);
    if (csvFile.is_open()) {
        // read data line by line
        std::string line;
        while (getline(csvFile, line)) {
            std::vector<std::string> currLine; // all the values from current line
            int pos = 0;
            std::string token;
            while ((pos = line.find(",")) != std::string::npos) {
                token = line.substr(0, pos);
                currLine.push_back(token);
                line.erase(0, pos + 1);
            }
            currLine.push_back(line);

            std::vector<double> currFeature; // all the values except the first one from current line
            if (currLine.size() != 0) {
                classNamesDB.push_back(currLine[0]);
                for (int i = 1; i < currLine.size(); i++) {
                    currFeature.push_back(stod(currLine[i]));
                }
                featuresDB.push_back(currFeature);
            }
        }
    }
}

std::string getClassName(char c) {
    std::map<char, std::string> myMap{
            {'p', "pen"}, {'s', "sharpener"}, {'h', "phone"}, {'g', "glasses"},
            {'n', "pen drive"}, {'c', "card"}, {'b', "book"}, {'t', "tomato"},
            {'o', "coin"}, {'w', "straw"},
    };
    return myMap[c];
}

std::string classifier(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature) {
    double THRESHOLD = 0.15;
    double distance = DBL_MAX;
    std::string className = " ";
    for (int i = 0; i < featureVectors.size(); i++) { // loop the known features to get the closed one
        std::vector<double> dbFeature = featureVectors[i];
       std:: string dbClassName = classNames[i];
        double curDistance = euclideanDistance(dbFeature, currentFeature);
        if (curDistance < distance && curDistance < THRESHOLD) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

double euclideanDistance(std::vector<double> features1, std::vector<double> features2) {
    double sum1 = 0, sum2 = 0, sumDifference=0;
    for (int i = 0; i < features1.size(); i++) {
        sumDifference += (features1[i] - features2[i]) * (features1[i] - features2[i]);
        sum1 += features1[i] * features1[i];
        sum2 += features2[i] * features2[i];
    }
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));
}

std::string classifierKNN(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature, int K) {
    double THRESHOLD = 0.15;
    // compute the distances of current feature vector with all the feature vectors in DB
    std::vector<double> distances;
    for (int i = 0; i < featureVectors.size(); i++) {
        std::vector<double> dbFeature = featureVectors[i];
        double distance = euclideanDistance(dbFeature, currentFeature);
        if (distance < THRESHOLD) {
            distances.push_back(distance);
        }
    }

    std::string className = " ";
    if (distances.size() > 0) {
        // sort the distances in ascending order
        std::vector<int> sortedIdx;
        cv::sortIdx(distances, sortedIdx, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

        // get the first K class name, and count the number of each name
        std::vector<std::string> firstKNames;
        int s = sortedIdx.size();
        std::map<std::string, int> nameCount;
        int range = std::min(s, K); // if less than K classnames, get all of them
        for (int i = 0; i < range; i++) {
            std::string name = classNames[sortedIdx[i]];
            if (nameCount.find(name) != nameCount.end()) {
                nameCount[name]++;
            }
            else {
                nameCount[name] = 1;
            }
        }

        // get the class name that appear the most times in the K nearest neighbors
        int count = 0;
        for (std::map<std::string, int>::iterator it = nameCount.begin(); it != nameCount.end(); it++) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
        }
    }
    return className;
}

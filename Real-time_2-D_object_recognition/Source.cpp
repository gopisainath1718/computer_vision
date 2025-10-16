#include "Header.h"

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cout << "Wrong input." << std::endl;
        exit(-1);
    }

    std::vector<std::string> classNamesDB;
    std::vector<std::vector<double>> featuresDB;

    bool training = false;

    cv::VideoCapture* capdev;
    capdev = new cv::VideoCapture(0);

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::namedWindow("video", 1); // identifies a window

    int n = 256;
    cv::Mat original_frame, copy, dilate_frame, erode_frame, kernel;
    
    loadFromCSV(argv[1], classNamesDB, featuresDB);

    while (true) {
        *capdev >> original_frame;
        if (original_frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        
        char key = cv::waitKey(10); // see if there is a waiting keystroke for the video

        // switch between training mode and inference mode
        if (key == 't') {
            training = !training;
            if (training)
                std::cout << "Training Mode" << std::endl;
            else
                std::cout << "Inference Mode" << std::endl;
        }

        original_frame.copyTo(copy);

        cv::cvtColor(copy, copy, cv::COLOR_BGR2GRAY);

        cv::Mat threshold_frame = threshold(copy);
        threshold_frame.copyTo(erode_frame);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(25, 25));
        morphologyEx(threshold_frame, erode_frame, cv::MORPH_CLOSE, kernel);

        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(erode_frame, labels, stats, centroids, 8);

        std::vector<int> counts;
        for (int i = 0; i < n; i++) {
            int count = 0;
            cv::Mat region_mask = (labels == i);
            for (int i = 0; i < labels.rows; i++) {
                uchar* ptr = region_mask.ptr<uchar>(i);
                for (int j = 0; j < labels.cols; j++) {
                    if (ptr[j] != 0)
                        count++;
                }
            }
            counts.push_back(count);
            //std::cout << i << " = " << counts[i] << "\n";
        }
        sort(counts.begin(), counts.end(), std::greater<int>());

        for (int i = 0; i < counts.size(); i++) {
            if (counts[i] >= 2000) {
                n = i + 1;
                if (n > 3)n = 3;
            }
        }
        for (int i = 0; i < labels.rows; i++) {
            uchar* ptr = labels.ptr<uchar>(i);
            for (int j = 0; j < labels.cols; j++) {
                if (ptr[j] > n)
                    ptr[j] = 0;
            }
        }

        cv::Mat out = color_segmentation(labels, n);
        std::cout<<"n= " << n << "\n";

        cv::Mat region_mask = (labels == 1);

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
        float alpha = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);
        drawLine(original_frame, centroid_x, centroid_y, alpha, cv::Scalar(0, 0, 255));
        drawBoundingBox(original_frame, erode_frame);

        if (training) {
            std::cout << "Input the class for this object." << std::endl;
            //namedWindow("Current Region", cv::WINDOW_AUTOSIZE);
            //cv::imshow("Current Region", region);

            // ask the user for a class name
            std::cout << "Input the class for this object." << std::endl;
            // get the code for each class name
            char k = cv::waitKey(0);
            std::string className = getClassName(k); //see the function for a detailed mapping

            // update the DB
            featuresDB.push_back(huMoments);
            classNamesDB.push_back(className);
            training = false;
        }
        else {
            std::string className;
            if (!strcmp(argv[2], "n")) { // nearest neighbor
                className = classifier(featuresDB, classNamesDB, huMoments);
            }
            else if (!strcmp(argv[2], "k")) { // KNN
                className = classifierKNN(featuresDB, classNamesDB, huMoments, 5);
            }
            // overlay classname to the video
            cv::putText(original_frame, className, cv::Point(centroid_x, centroid_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("video", original_frame);
        cv::imshow("threshold_video", threshold_frame);
        cv::imshow("out_video", out);
        cv::imshow("mask_video", erode_frame);

        char ch = cv::waitKey(10);
        if (ch == 'q') {
            writeToCSV(argv[1], classNamesDB, featuresDB);
            cv::destroyAllWindows();
            break;
        }
    }
     delete capdev;
	return 0;
}

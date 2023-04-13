#include"Header.h"

int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;
    cv::VideoCapture* img;

    // open the video device
    capdev = new cv::VideoCapture(0);

    img = new cv::VideoCapture("D:/movies/Jersey-2019.mp4");
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, video_frame;
    cv::Vec<float, 5> dist_coef(0, 0, 0, 0, 0);
    cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);

    readCSV("parameters.csv", camera_matrix, dist_coef);
    //std::cout << camera_matrix << "\n";
    //for (int i = 0; i < 5; i++)
    //    std::cout << dist_coef[i] << " ";

    cv::Size patternSize = cv::Size(9, 6);
    cv::Mat rvecs, tvecs;
    int flags = cv::SOLVEPNP_ITERATIVE;

    float m = 1,c;
    int count = 0;

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream

        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        std::vector<cv::Vec3f> point_set;
        
        std::vector<cv::Point2f> corner_set;
        std::vector<cv::Point2f> corner_set_1;
        bool patternfound = corners(frame, corner_set);

        for (int y = 0; y > -6; y--) {
            for (int x = 0; x < 9; x++) {
                //cv::Vec3f point = (x, y, 0);
                //std::cout << x << "," << y<<"\n";
                point_set.push_back(cv::Vec3f(x, y, 0));
            }
        }

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }  
        switch (key) {
        case 'a':
            count = 1;  break;
        case 'g':
            count = 2;  break;
        case 'm':
            count = 3;  break;
        case 'v':
            count = 4; break;
        case 'd':
            count = 5; break;
        case 'e':
            count = 6; break;
        }

        if (patternfound) {
            cv::solvePnP(point_set, corner_set, camera_matrix, dist_coef, rvecs, tvecs, false, flags);
            std::cout << "rvecs=\n" << rvecs << "\ntvecs=\n" << tvecs<<"\n";

            if (count == 1) {
                axes(frame, rvecs, tvecs, camera_matrix, dist_coef, corner_set);
            }
            if (count == 2) {
                //image(frame, corner_set);
                letter_g(frame, rvecs, tvecs, camera_matrix, dist_coef);
            }
            if (count == 3) {
                if (m <= 6)
                    motion(frame, rvecs, tvecs, camera_matrix, dist_coef, m);
                else
                    m = 1;
            }
            if (count == 4) {
                *img >> video_frame;
                video(frame, video_frame, corner_set);
            }

            if (count == 5) {
                project_circle(frame, rvecs, tvecs, camera_matrix);
            }

            if (count == 6)
                project_pyramid(frame, rvecs, tvecs, camera_matrix);
        }


        cv::imshow("Video", frame);
        
        if (key == 'h') {
            int thresh = 150;
            int blockSize = 2;
            int apertureSize = 3;
            double k = 0.04;
            cv::Mat src_gray;
            cv::Mat dst = cv::Mat::zeros(frame.size(), CV_32FC1);
            cv::cvtColor(frame, src_gray, cv::COLOR_BGR2GRAY);
            cv::cornerHarris(src_gray, dst, blockSize, apertureSize, k);
            cv::Mat dst_norm, dst_norm_scaled;
            cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::convertScaleAbs(dst_norm, dst_norm_scaled);
            for (int i = 0; i < dst_norm.rows; i++)
            {
                for (int j = 0; j < dst_norm.cols; j++)
                {
                    if ((int)dst_norm.at<float>(i, j) > thresh)
                    {
                        circle(dst_norm_scaled, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
                    }
                }
            }
            cv::imshow("corners_window", dst_norm_scaled);
        }
        }



    delete capdev;
    return(0);
}
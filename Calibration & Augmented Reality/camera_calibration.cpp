//#include "Header.h"
//
//int main(int argc, char* argv[]) {
//
//    cv::namedWindow("Video", 1); // identifies a window
//    cv::Mat frame;
//    
// 
//    std::vector<std::vector<cv::Vec3f> > point_list;
//    std::vector<std::vector<cv::Point2f> > corner_list;
//    //cv::Size subPixWinSize(10, 10), winSize(31, 31);
//    //cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
//    int img_count = 0;
//
//    cv::VideoCapture* capdev;
//    // open the video device
//    capdev = new cv::VideoCapture(0);
//    if (!capdev->isOpened()) {
//        printf("Unable to open video device\n");
//        return(-1);
//    }
//
//    // get some properties of the image
//    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
//        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
//    printf("Expected size: %d %d\n", refS.width, refS.height);
//    
//    while(1) {
//        
//        *capdev >> frame; // get a new frame from the camera, treat as a stream
//        if (frame.empty()) {
//            printf("frame is empty\n");
//            break;
//        }
//        std::vector<cv::Vec3f> point_set;
//        std::vector<cv::Point2f> corner_set;
//        bool patternfound = corners(frame, corner_set);
//
//
//
//        cv::imshow("Video", frame);
//        
//        char key = cv::waitKey(10);     // see if there is a waiting keystroke
//        if (key == 's') {
//            points_3d(patternfound, img_count, frame, corner_set, corner_list, point_set, point_list);
//        }
//
//        calibration(frame,img_count, point_list, corner_list);
//
//        if (key == 'q') {
//            break;
//        }
//    }
//
//    delete capdev;
//    return(0);
//}
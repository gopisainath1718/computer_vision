#include"Header.h"
#include<iostream>
#include<opencv2/opencv.hpp>

int i = 0;      // counter variable for saved images
int count = 0;      // counter variable for applying filters

void func(cv::Mat frame, int key) {
    switch (key) {
    case 'o':       //for original video.
        count = 0;
        break;
    case 'g':       //for greyscale video.          
        count = 1;
        break;
    case 'h':       //for alternate greyscale video.
        count = 2;
        break;
    case 'b':       //for blurred video.
        count = 3;
        break;
    case 'x':       //for X-sobel filter.
        count = 4;
        break;
    case 'y':       //for Y-sobel filter.
        count = 5;
        break;
    case 'm':       //for gradient magnitude video.
        count = 6;
        break;
    case 'i':       //for quantized video.
        count = 7;
        break;
    case 'c':       //for cartoon video.
        count = 8;
        break;
    case 'a':       //for alternate blurred video.
        count = 9;
        break;
    }
    if (key == 's') {    //for saving the image.
        i++;
        std::string name = "saved_image-" + std::to_string(i) + ".jpg";
        cv::imwrite(name, frame);   
        std::cout << "Image" << i << "saved.\n";
    }
    if (count == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);   // converts into grey scale frame.
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);

        }
    }
    else if (count == 2) {
        cv::Mat alternate_frame(480, 640, CV_8UC3);
        frame = greyscale(frame, alternate_frame);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);

        }
    }
    else if (count == 3) {
        cv::Mat dst;
        frame.copyTo(dst);
        frame = blur5x5(frame, dst);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);

        }
    }
    else if (count == 4) {
        cv::Mat dst(frame.rows, frame.cols/2, CV_16SC3);
        frame = sobelX3x3(frame, dst);   // function call
        //dst.convertTo(dst, CV_16SC3);
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);
        }
    }
    else if (count == 5) {
        cv::Mat dst(frame.rows, frame.cols/2, CV_16SC3);
        frame = sobelY3x3(frame, dst);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);
        }
    }
    else if (count == 6) {
        cv::Mat dst(frame.rows, frame.cols, CV_8UC3);
        cv::Mat sx = sobelX3x3(frame, dst);
        cv::Mat sy = sobelY3x3(frame, dst);
        //dst.convertTo(dst, CV_16SC3);
        frame = magnitude(sx, sy, dst);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);
        }
    }
    else if (count == 7) {
        int levels = 15;
        cv::Mat alternate_frame;
        frame.copyTo(alternate_frame);
        frame = blurQuantize(frame, alternate_frame, levels);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);
        }
    }
    else if (count == 8) {
        int levels = 8;
        int magThreshold = 40;
        cv::Mat dst;
        frame.copyTo(dst);
        frame = cartoon(frame, dst, levels, magThreshold);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);
        }
    }
    else if (count == 9) {
        cv::Mat dst;
        frame.copyTo(dst);
        frame = mean3x3(frame, dst);   // function call
        if (key == 's') {
            std::string name = "saved_image-" + std::to_string(i) + ".jpg";
            cv::imwrite(name, frame);

        }
    }
    cv::imshow("Video", frame);   // outputs the frame.
}

int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame[2];

    bool flag = true;

    std::cout << "Welcome to the real time filtering application." << std::endl;
    std::cout << "You can press the keys in any order to get the output.\n";
    std::cout << "You can press 'f' anytime to get the flipped video.\n";
    std::cout << "If you liked any filter feel free to press 's' and save the image.\n";
    std::cout << "Finally if you like to take a leave press 'q' to quit.\n\n";

    std::cout << "g - For getting a grey scale version press 'g'.\n";
    std::cout << "h - Didn't like that..? press 'h' for an alternate grey scale version.\n";
    std::cout << "b - Want a blurred video..? press 'b'.\n";
    std::cout << "a - Didn't like that..? press 'a' for an alternate blurred video.\n";
    std::cout << "x - Press 'x' for a X-sobel filter.\n";
    std::cout << "y - Press 'y' for a Y-sobel filter.\n";
    std::cout << "m - Do you wanna see gradient magnitude video press 'm'.\n";
    std::cout << "i - What if we quantize the video..? press 'i' and have a look.\n";
    std::cout << "c - what about cartoons..? press 'c' and see yourself as an cartoon.\n";
    std::cout << "o - Missing the original one..? press 'o' to see yourself again.\n\n";


    while (true) {
        *capdev >> frame[0];   // get a new frame from the camera, treat as a stream

        if (frame[0].empty()) {
            printf("frame is empty\n");
            break;
        }
        cv::flip(frame[0], frame[1], 1);   // flipping the frame
        char key = cv::waitKey(10);

        if (key == 'f') {
            if (flag == false) flag = true; // changing flag value
            else flag = false;
        }

        // if the flag is true then filters are applied on original frame.
        // if the flag is flase then filters are applied on flipped frame. 
        if (flag == true) {
            if (key == 'q')
                break;
            func(frame[0], key);
        }
        else if (flag == false) {
            if (key == 'q')
                break;
            func(frame[1], key);
        }
    }
    delete capdev;
    return(0);
}

/*
Bruce A. Maxwell

CS 5330 Computer Vision
Spring 2021

CPP functions for reading CSV files with a specific format
- first column is a string containing a filename or path
- every other column is a number

The function returns a std::vector of char* for the filenames and a 2D std::vector of floats for the data
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"

/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */
int getstring(FILE* fp, char os[]) {
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        // printf("%c", ch ); // uncomment for debugging
        os[p] = ch;
        p++;
    }
    // printf("\n"); // uncomment for debugging
    os[p] = '\0';

    return(eol); // return true if eol
}

int getint(FILE* fp, int* v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0'; // terminator
    *v = atoi(s);

    return(eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE* fp, float* v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0'; // terminator
    *v = atof(s);

    return(eol); // return true if eol
}

/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file) {
    char buffer[256];
    char mode[8];
    FILE* fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }

    fp = fopen(filename, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", filename);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    strcpy(buffer, image_filename);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
    for (int i = 0; i < image_data.size(); i++) {
        char tmp[256];
        sprintf_s(tmp, ",%.4f", image_data[i]);
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp); // EOL

    fclose(fp);

    return(0);
}

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<std::vector<float>>& data, int echo_file) {
    FILE* fp;
    float fval;
    char img_file[256];

    fp = fopen(filename, "r");
    if (!fp) {
        printf("Unable to open feature file\n");
        return(-1);
    }

    printf("Reading %s\n", filename);
    for (;;) {
        std::vector<float> dvec;


        // read the filename
        if (getstring(fp, img_file)) {
            break;
        }
        // printf("Evaluting %s\n", filename);

        // read the whole feature file into memory
        for (;;) {
            // get next feature
            float eol = getfloat(fp, &fval);
            dvec.push_back(fval);
            if (eol) break;
        }
        // printf("read %lu features\n", dvec.size() );

        data.push_back(dvec);

        char* fname = new char[strlen(img_file) + 1];
        strcpy(fname, img_file);
        filenames.push_back(fname);
    }
    fclose(fp);
    printf("Finished reading CSV file\n");

    if (echo_file) {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[i].size(); j++) {
                printf("%.4f  ", data[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return(0);
}

std::vector<float> baseline_feature(char* filepath) {
    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img = cv::imread(filepath);
    
    //creating the reagion lof interest
    cv::Rect roi((img.rows / 2) - (9/2), (img.cols/2)-(9/2), 9, 9);
    cv::Mat f = img(roi);

    //creating the feature vector
    for (int i = 0;i<f.rows;i++) {
        cv::Vec3b* a = f.ptr<cv::Vec3b>(i);
        for (int j = 0;j<f.cols;j++) {
            for (int c= 0;c<3;c++) {
                f_vector.push_back(a[j][c]);
            }
        }
    }
    return f_vector;
}

std::vector<float> histogram_feature(char* filepath) {
    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img = cv::imread(filepath);

    //creating the blue and green 2D histogram
    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0;i<img.rows;i++) {
        cv::Vec3b* img_ptr = img.ptr<cv::Vec3b>(i);
        for (int j = 0;j<img.cols;j++) {
            int b_index = int((float(img_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img_ptr[j][1]) / 256) * 16);

            int* bg_hist_ptr = bg_hist.ptr<int>(b_index);
            bg_hist_ptr[g_index] = bg_hist_ptr[g_index] + 1;
        }
    }

    //creating the feature vector
    for (int i = 0; i< bg_hist.rows; i++) {
        int* bg_hist_ptr = bg_hist.ptr<int>(i);
        for (int j= 0;j<bg_hist.cols;j++) {
            f_vector.push_back(bg_hist_ptr[j]);
        }
    }
    return f_vector;
}

std::vector<float> multi_histogram_feature(char* filepath) {
    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img = cv::imread(filepath);
    const int size[] = { 8, 8, 8 };
    cv::Mat bgr_hist(3, size, CV_32SC1, cv::Scalar(0));
    cv::Mat bgr_inner_hist(3, size, CV_32SC1, cv::Scalar(0));

    //creating the region of interest
    cv::Rect roi((img.rows / 2) - (250 / 2), (img.cols / 2) - (350 / 2), 350, 250);
    cv::Mat inner_img = img(roi);

    //creating the BGR 3D histogram from total image
    for (int i = 0;i<img.rows;i++) {
        cv::Vec3b* img_ptr = img.ptr<cv::Vec3b>(i);
        for (int j = 0;j<img.cols;j++) {
            int b_index = int((float(img_ptr[j][0]) / 256) * 8);
            int g_index = int((float(img_ptr[j][1]) / 256) * 8);
            int r_index = int((float(img_ptr[j][2]) / 256) * 8);

            int* bgr_hist_ptr = bgr_hist.ptr<int>(b_index);
            bgr_hist_ptr[8 * (g_index)+r_index] += 1;
        }
    }

    //creating the BGR 3D histogram from inner image
    for (int i = 0; i < inner_img.rows; i++) {
        cv::Vec3b* inner_img_ptr = inner_img.ptr<cv::Vec3b>(i);
        for (int j = 0; j < inner_img.cols; j++) {
            int b_index = int((float(inner_img_ptr[j][0]) / 256) * 8);
            int g_index = int((float(inner_img_ptr[j][1]) / 256) * 8);
            int r_index = int((float(inner_img_ptr[j][2]) / 256) * 8);

            int* bgr_inner_hist_ptr = bgr_inner_hist.ptr<int>(b_index);
            bgr_inner_hist_ptr[8 * (g_index)+r_index] += 1;
        }
    }

    //creating the feature vector
    for (int i = 0; i <8; i++) {
        int* bgr_hist_ptr = bgr_hist.ptr<int>(i);
        for (int j = 0; j < 8; j++) {
            for (int c = 0;c <8;c++) {
                f_vector.push_back(bgr_hist_ptr[8*j+c]);
            }
        }
    }
    f_vector.push_back(0.1);      //pushing something random to distinguish both histograms 

    for (int i = 0; i <8; i++) {
        int* bgr_inner_hist_ptr = bgr_inner_hist.ptr<int>(i);
        for (int j = 0; j < 8; j++) {
            for (int c = 0; c <8; c++) {
                f_vector.push_back(bgr_inner_hist_ptr[8*j+c]);
            }
        }
    }
    return f_vector;
}

std::vector<float> TC_feature(char* filepath) {
    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img = cv::imread(filepath);

    //creating the BG 2D histogram for the image
    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0; i < img.rows; i++) {
        cv::Vec3b* img_ptr = img.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img.cols; j++) {
            int b_index = int((float(img_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img_ptr[j][1]) / 256) * 16);

            int* bg_hist_ptr = bg_hist.ptr<int>(b_index);
            bg_hist_ptr[g_index] = bg_hist_ptr[g_index] + 1;
        }
    }

    cv::Mat gradX, gradY, gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);        //converting image into grayscale
    // Sobel filter for gradient in X direction
    Sobel(gray, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    // Sobel filter for gradient in Y direction
    Sobel(gray, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    //creating the region of interest
    cv::Rect r1(0, 0, gradX.cols/2, gradX.rows/2);
    cv::Rect r2(0, 0, gradY.cols/2, gradY.rows/ 2);
    cv::Mat roi1 = gradX(r1);
    cv::Mat roi2 = gradY(r2);

    //creating 2D histogram with magnitude and orientation information
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0; i < roi1.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1.cols; j++) {
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);       //dividing into 8 bins 
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);        //dividing into 8 bins

            cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(mag_index);
            hist_ptr[orientation_index][0]++;
        }
    }

    //creating the feature vector
    for (int i = 0; i < bg_hist.rows; i++) {
        cv::Vec3b* bg_hist_ptr = bg_hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < bg_hist.cols; j++) {
            f_vector.push_back(bg_hist_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
  //while (1) {
  //  cv::imshow("img1", gradX);
  //  cv::imshow("img2", gradY);
  //  cv::imshow("roi1_q1", roi1);
  //  cv::imshow("roi2_q1", roi2);      //for debugging purpose
  //  if (cv::waitKey(0) == 'q')
  //      break;
  //}
  //  cv::destroyAllWindows();
    return f_vector;
}

std::vector<float> TC_feature_1(char* filepath) {
    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img = cv::imread(filepath);

    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);

    //creating the BG 2D histogram
    for (int i = 0; i < img.rows; i++) {
        cv::Vec3b* img_ptr = img.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img.cols; j++) {
            int b_index = int((float(img_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img_ptr[j][1]) / 256) * 16);

            int* bg_hist_ptr = bg_hist.ptr<int>(b_index);
            bg_hist_ptr[g_index] = bg_hist_ptr[g_index] + 1;
        }
    }

    cv::Mat gradX, gradY, gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);        //converting image into grayscale

    // Sobel filter for gradient in X direction
    Sobel(gray, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    // Sobel filter for gradient in Y direction
    Sobel(gray, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    //diving the image into 4 quadrants and creating the region of interests 
    cv::Rect r1_q1(0, 0, gradX.cols / 2, gradX.rows / 2);
    cv::Rect r2_q1(0, 0, gradY.cols / 2, gradY.rows / 2);
    cv::Rect r1_q2(gradX.cols / 2, 0, gradX.cols / 2, gradX.rows / 2);
    cv::Rect r2_q2(gradX.cols / 2, 0, gradY.cols / 2, gradY.rows / 2);
    cv::Rect r1_q3(0, gradX.rows / 2, gradX.cols / 2, gradX.rows / 2);
    cv::Rect r2_q3(0, gradX.rows / 2, gradY.cols / 2, gradY.rows / 2);
    cv::Rect r1_q4(gradX.cols / 2, gradX.rows / 2, gradX.cols / 2, gradX.rows / 2);
    cv::Rect r2_q4(gradX.cols / 2, gradX.rows / 2, gradY.cols / 2, gradY.rows / 2);
    cv::Mat roi1_q1 = gradX(r1_q1);
    cv::Mat roi2_q1 = gradY(r2_q1);
    cv::Mat roi1_q2 = gradX(r1_q2);
    cv::Mat roi2_q2 = gradY(r2_q2);
    cv::Mat roi1_q3 = gradX(r1_q3);
    cv::Mat roi2_q3 = gradY(r2_q3);
    cv::Mat roi1_q4 = gradX(r1_q4);
    cv::Mat roi2_q4 = gradY(r2_q4);

    //creating four 2D histograms with magnitude and orientation
    cv::Mat hist_q1 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q2 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q3 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q4 = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0; i < roi1_q1.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q1.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q1.cols; j++) {
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_q1_ptr = hist_q1.ptr<cv::Vec3b>(mag_index);
            hist_q1_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q2.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q2.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q2.cols; j++) {

            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_q2_ptr = hist_q2.ptr<cv::Vec3b>(mag_index);
            hist_q2_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q3.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q3.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q3.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q3.cols; j++) {

            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_q3_ptr = hist_q3.ptr<cv::Vec3b>(mag_index);
            hist_q3_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q4.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q4.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q4.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q4.cols; j++) {

            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_q4_ptr = hist_q4.ptr<cv::Vec3b>(mag_index);
            hist_q4_ptr[orientation_index][0]++;
        }
    }

    //creating the feature vector
    for (int i = 0; i < bg_hist.rows; i++) {
        cv::Vec3b* bg_hist_ptr = bg_hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < bg_hist.cols; j++) {
            f_vector.push_back(bg_hist_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist_q1.rows; i++) {
        cv::Vec3b* hist_q1_ptr = hist_q1.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q1.cols; j++) {
            f_vector.push_back(hist_q1_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist_q2.rows; i++) {
        cv::Vec3b* hist_q2_ptr = hist_q2.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q2.cols; j++) {
            f_vector.push_back(hist_q2_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist_q3.rows; i++) {
        cv::Vec3b* hist_q3_ptr = hist_q3.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q3.cols; j++) {
            f_vector.push_back(hist_q3_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist_q4.rows; i++) {
        cv::Vec3b* hist_q4_ptr = hist_q4.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q4.cols; j++) {
            f_vector.push_back(hist_q4_ptr[j][0]);
        }
    }
    return f_vector;
}

std::vector<float> extension(char* filepath){
    std::vector<float> f_vector;

    //readin the image path
    cv::Mat img = cv::imread(filepath);
    cv::Mat dst1, dst, dst2;
    img.copyTo(dst);
    img.copyTo(dst1);
    img.copyTo(dst2);

    //creating the BG histogram
    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0; i < img.rows; i++) {
        cv::Vec3b* img_ptr = img.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img.cols; j++) {
            int b_index = int((float(img_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img_ptr[j][1]) / 256) * 16);

            int* bg_hist_ptr = bg_hist.ptr<int>(b_index);
            bg_hist_ptr[g_index] = bg_hist_ptr[g_index] + 1;
        }
    }

    //applying the L5E5 filter
    for (int i = 2; i < img.rows - 3; i++) {
        // create row pointers for accessing the rows
        cv::Vec3b* p2_ptr = img.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* p1_ptr = img.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
        cv::Vec3b* n1_ptr = img.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* n2_ptr = img.ptr<cv::Vec3b>(i + 2);
        cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                // applying convolution in horizontal direction
                dst_ptr[j][c] = 1 * p2_ptr[j][c] + 4 * p1_ptr[j][c] + 6 * ptr[j][c] +
                    4 * n1_ptr[j][c] + 1 * n2_ptr[j][c];
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
                dst1_ptr[k][a] = 1 * pptr[k - 2][a] + 2 * pptr[k - 1][a] +
                    (-2)* pptr[k + 1][a] + (-1) * pptr[k + 2][a];
            }
        }
    }

    //applying the L5E5 transpose filter
    for (int i = 2; i < img.rows - 3; i++) {
        // create row pointers for accessing the rows
        cv::Vec3b* p2_ptr = img.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* p1_ptr = img.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
        cv::Vec3b* n1_ptr = img.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* n2_ptr = img.ptr<cv::Vec3b>(i + 2);
        cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                // applying convolution in horizontal direction
                dst_ptr[j][c] = 1 * p2_ptr[j][c] + 2 * p1_ptr[j][c] + 0 * ptr[j][c] +
                    (-2) * n1_ptr[j][c] + (-1) * n2_ptr[j][c];
            }
        }
    }
    for (int i = 0; i < dst.rows - 1; i++) {
        // create row pointers for accessing the rows
        cv::Vec3b* pptr = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b* dst2_ptr = dst2.ptr<cv::Vec3b>(i);

        for (int k = 2; k < dst.cols - 3; k++) {
            for (int a = 0; a < 3; a++) {
                // applying convolution in vertical direction
                dst2_ptr[k][a] = 1 * pptr[k - 2][a] + 4 * pptr[k - 1][a] + 6*pptr[k][a]+
                    4 * pptr[k + 1][a] + 1 * pptr[k + 2][a];
            }
        }
    }
    
    //creating the region of interests
    cv::Rect r1((dst1.cols / 2)-50, 0, (dst1.cols / 2)-50, dst1.rows / 2);
    cv::Rect r2((dst2.cols / 2)-50, 0, (dst2.cols / 2)-50, dst2.rows / 2);
    cv::Mat roi1 =dst1(r1);
    cv::Mat roi2 = dst2(r2);

    //creating the 2D histograms using magnitude and orientation
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32SC1);
    for (int i = 0; i < roi1.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1.cols; j++) {
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(mag_index);
            hist_ptr[orientation_index][0]++;
        }
    }

    //creating the feature vector
    for (int i = 0; i < bg_hist.rows; i++) {
        int* bg_hist_ptr = bg_hist.ptr<int>(i);
        for (int j = 0; j < bg_hist.cols; j++) {
            f_vector.push_back(bg_hist_ptr[j]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
    return f_vector;
}

std::vector<float> extension_1(char* filepath) {

    std::vector<float> f_vector;
    cv::Mat img = cv::imread(filepath);
    cv::Mat dst1, dst, dst2;
    img.copyTo(dst);
    img.copyTo(dst1);
    img.copyTo(dst2);
    cv::Mat hist_q1 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q2 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q3 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist_q4 = cv::Mat::zeros(16, 16, CV_32SC1);

    for (int i = 2; i < img.rows - 3; i++) {
        // create row pointers for accessing the rows
        cv::Vec3b* p2_ptr = img.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* p1_ptr = img.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
        cv::Vec3b* n1_ptr = img.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* n2_ptr = img.ptr<cv::Vec3b>(i + 2);
        cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                // applying convolution in horizontal direction
                dst_ptr[j][c] = 1 * p2_ptr[j][c] + 4 * p1_ptr[j][c] + 6 * ptr[j][c] +
                    4 * n1_ptr[j][c] + 1 * n2_ptr[j][c];
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
                dst1_ptr[k][a] = 1 * pptr[k - 2][a] + 2 * pptr[k - 1][a] +
                    (-2) * pptr[k + 1][a] + (-1) * pptr[k + 2][a];
            }
        }
    }
    for (int i = 2; i < img.rows - 3; i++) {
        // create row pointers for accessing the rows
        cv::Vec3b* p2_ptr = img.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* p1_ptr = img.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
        cv::Vec3b* n1_ptr = img.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* n2_ptr = img.ptr<cv::Vec3b>(i + 2);
        cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                // applying convolution in horizontal direction
                dst_ptr[j][c] = 1 * p2_ptr[j][c] + 2 * p1_ptr[j][c] + 0 * ptr[j][c] +
                    (-2) * n1_ptr[j][c] + (-1) * n2_ptr[j][c];
            }
        }
    }
    for (int i = 0; i < dst.rows - 1; i++) {

        // create row pointers for accessing the rows
        cv::Vec3b* pptr = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b* dst2_ptr = dst2.ptr<cv::Vec3b>(i);

        for (int k = 2; k < dst.cols - 3; k++) {
            for (int a = 0; a < 3; a++) {

                // applying convolution in vertical direction
                dst2_ptr[k][a] = 1 * pptr[k - 2][a] + 4 * pptr[k - 1][a] + 6 * pptr[k][a] +
                    4 * pptr[k + 1][a] + 1 * pptr[k + 2][a];
            }
        }
    }
    cv::Rect r1_q1(0, 0, dst1.cols / 2, dst1.rows / 2);
    cv::Rect r2_q1(0, 0, dst2.cols / 2, dst2.rows / 2);
    cv::Rect r1_q2(dst1.cols / 2, 0, dst1.cols / 2, dst1.rows / 2);
    cv::Rect r2_q2(dst2.cols / 2, 0, dst2.cols / 2, dst2.rows / 2);
    cv::Rect r1_q3(0, dst1.rows / 2, dst1.cols / 2, dst1.rows / 2);
    cv::Rect r2_q3(0, dst2.rows / 2, dst2.cols / 2, dst2.rows / 2);
    cv::Rect r1_q4(dst1.cols / 2, dst1.rows / 2, dst1.cols / 2, dst1.rows / 2);
    cv::Rect r2_q4(dst2.cols / 2, dst2.rows / 2, dst2.cols / 2, dst2.rows / 2);
    cv::Mat roi1_q1 = dst1(r1_q1);
    cv::Mat roi2_q1 = dst2(r2_q1);
    cv::Mat roi1_q2 = dst1(r1_q2);
    cv::Mat roi2_q2 = dst2(r2_q2);
    cv::Mat roi1_q3 = dst1(r1_q3);
    cv::Mat roi2_q3 = dst2(r2_q3);
    cv::Mat roi1_q4 = dst1(r1_q4);
    cv::Mat roi2_q4 = dst2(r2_q4);

    for (int i = 0; i < roi1_q1.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q1.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q1.cols; j++) {

            //std::cout << x<<" " << y << " ";
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);
            //std::cout << mag_index<<" "<<orientation_index<<" ";

            cv::Vec3b* hist_q1_ptr = hist_q1.ptr<cv::Vec3b>(mag_index);
            hist_q1_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q2.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q2.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q2.cols; j++) {

            //std::cout << x<<" " << y << " ";
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);
            //std::cout << mag_index<<" "<<orientation_index<<" ";

            cv::Vec3b* hist_q2_ptr = hist_q2.ptr<cv::Vec3b>(mag_index);
            hist_q2_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q3.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q3.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q3.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q3.cols; j++) {

            //std::cout << x<<" " << y << " ";
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);
            //std::cout << mag_index<<" "<<orientation_index<<" ";

            cv::Vec3b* hist_q3_ptr = hist_q3.ptr<cv::Vec3b>(mag_index);
            hist_q3_ptr[orientation_index][0]++;
        }
    }
    for (int i = 0; i < roi1_q4.rows; i++) {
        cv::Vec3b* x_ptr1 = roi1_q4.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = roi2_q4.ptr<cv::Vec3b>(i);

        for (int j = 0; j < roi1_q4.cols; j++) {

            //std::cout << x<<" " << y << " ";
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);
            //std::cout << mag_index<<" "<<orientation_index<<" ";

            cv::Vec3b* hist_q4_ptr = hist_q4.ptr<cv::Vec3b>(mag_index);
            hist_q4_ptr[orientation_index][0]++;
        }
    }

    for (int i = 0; i < hist_q1.rows; i++) {
        cv::Vec3b* hist_q1_ptr = hist_q1.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q1.cols; j++) {
            f_vector.push_back(hist_q1_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);
    for (int i = 0; i < hist_q2.rows; i++) {
        cv::Vec3b* hist_q2_ptr = hist_q2.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q2.cols; j++) {
            f_vector.push_back(hist_q2_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);
    for (int i = 0; i < hist_q3.rows; i++) {
        cv::Vec3b* hist_q3_ptr = hist_q3.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q3.cols; j++) {
            f_vector.push_back(hist_q3_ptr[j][0]);
        }
    }
    f_vector.push_back(0.1);
    for (int i = 0; i < hist_q4.rows; i++) {
        cv::Vec3b* hist_q4_ptr = hist_q4.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist_q4.cols; j++) {
            f_vector.push_back(hist_q4_ptr[j][0]);
        }
    }
    return f_vector;
}

std::vector<float> custom_design_feature(char* filepath1, char* filepath2) {

    std::vector<float> f_vector;
    
    //reading the image paths
    cv::Mat img1 = cv::imread(filepath1);
    cv::Mat img2 = cv::imread(filepath2);

    //initializing the histograms
    cv::Mat hist1 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist2 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat bg_hist1 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat bg_hist2 = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);

    //creating the BG histogram
    for (int i = 0; i < img1.rows; i++) {
        cv::Vec3b* img1_ptr = img1.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img1.cols; j++) {
            int b_index = int((float(img1_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img1_ptr[j][1]) / 256) * 16);

            int* bg_hist1_ptr = bg_hist1.ptr<int>(b_index);
            bg_hist1_ptr[g_index] = bg_hist1_ptr[g_index] + 1;
        }
    }
    for (int i = 0; i < img2.rows; i++) {
        cv::Vec3b* img2_ptr = img2.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img2.cols; j++) {
            int b_index = int((float(img2_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img2_ptr[j][1]) / 256) * 16);

            int* bg_hist2_ptr = bg_hist2.ptr<int>(b_index);
            bg_hist2_ptr[g_index] = bg_hist2_ptr[g_index] + 1;
        }
    }

    cv::Mat gray1, gray2, sobel1, sobel2, grad_x1, grad_y1, grad_x2, grad_y2;

    //converting into grayscale
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // Sobel filter for gradient in X direction
    cv::Sobel(gray1, grad_x1, CV_64FC1, 1, 0, 3);
    // Sobel filter for gradient in Y direction
    cv::Sobel(gray1, grad_y1, CV_64FC1, 0, 1, 3);
    cv::convertScaleAbs(grad_x1, grad_x1);
    cv::convertScaleAbs(grad_y1, grad_y1);

    // Sobel filter for gradient in X direction
    cv::Sobel(gray2, grad_x2, CV_64FC1, 1, 0, 3);
    // Sobel filter for gradient in Y direction
    cv::Sobel(gray2, grad_y2, CV_64FC1, 0, 1, 3);
    cv::convertScaleAbs(grad_x2, grad_x2);
    cv::convertScaleAbs(grad_y2, grad_y2);

    //creating region of interests for image 1
    cv::Rect img_1_r1(grad_x1.cols / 4, grad_x1.rows / 4, grad_x1.cols / 2, grad_x1.rows / 2);
    cv::Rect img_1_r2(grad_x1.cols / 4, grad_x1.rows / 4, grad_y1.cols / 2, grad_y1.rows / 2);
    cv::Mat img_1_roi1 = grad_x1(img_1_r1);
    cv::Mat img_1_roi2 = grad_y1(img_1_r2);

    //creating region of interests for image 2
    cv::Rect img_2_r1(grad_y2.cols / 4, grad_x2.rows / 4, grad_x2.cols / 2, grad_x2.rows / 2);
    cv::Rect img_2_r2(grad_y2.cols / 4, grad_x2.rows / 4, grad_y2.cols / 2, grad_y2.rows / 2);
    cv::Mat img_2_roi1 = grad_x2(img_2_r1);
    cv::Mat img_2_roi2 = grad_y2(img_2_r2);

    //creating the histograms using magnitude and orientation
    for (int i = 0; i < img_1_roi1.rows; i++) {
        cv::Vec3b* x_ptr1 = img_1_roi1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = img_1_roi2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img_1_roi1.cols; j++) {
            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_ptr1 = hist1.ptr<cv::Vec3b>(mag_index);
            hist_ptr1[orientation_index][0]++;
        }
    }

    for (int i = 0; i < img_2_roi1.rows; i++) {
        cv::Vec3b* x_ptr2 = img_2_roi1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr2 = img_1_roi2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img_1_roi1.cols; j++) {

            int mag_index = int((std::sqrt(x_ptr2[j][0] * x_ptr2[j][0] + y_ptr2[j][0] * y_ptr2[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr2[j][0], x_ptr2[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_ptr2 = hist2.ptr<cv::Vec3b>(mag_index);
            hist_ptr2[orientation_index][0]++;
        }
    }

    //creating feature vector
    for (int i = 0; i < bg_hist.rows; i++) {
        int* bg_hist1_ptr = bg_hist1.ptr<int>(i);
        int* bg_hist2_ptr = bg_hist2.ptr<int>(i);
        int* bg_hist_ptr = bg_hist.ptr<int>(i);
        for (int j = 0; j < bg_hist.cols; j++) {
            bg_hist_ptr[j] = (bg_hist1_ptr[j] + bg_hist2_ptr[j])/2;
            f_vector.push_back(bg_hist_ptr[j]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist1.rows; i++) {
        cv::Vec3b* hist_ptr1 = hist1.ptr<cv::Vec3b>(i);
        cv::Vec3b* hist_ptr2 = hist2.ptr<cv::Vec3b>(i);
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist1.cols; j++) {
            hist_ptr[j][0] = (hist_ptr1[j][0] + hist_ptr2[j][0])/2;
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
//while (1) {
//  cv::imshow("img1", img_1_roi1);
//  cv::imshow("img2", img_1_roi2);
//  cv::imshow("roi1_q1", img_2_roi1);      //for debugging purpose
//  cv::imshow("roi2_q1", img_2_roi2);
//  if (cv::waitKey(0) == 'q')
//      break;
//}
//  cv::destroyAllWindows();
    return f_vector;
}

std::vector<float> custom_design_feature_1(char* filepath) {

    std::vector<float> f_vector;

    //reading the image path
    cv::Mat img1 = cv::imread(filepath);
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32SC1);
    cv::Mat bg_hist = cv::Mat::zeros(16, 16, CV_32SC1);

    //creating the BG histogram
    for (int i = 0; i < img1.rows; i++) {
        cv::Vec3b* img1_ptr = img1.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img1.cols; j++) {
            int b_index = int((float(img1_ptr[j][0]) / 256) * 16);
            int g_index = int((float(img1_ptr[j][1]) / 256) * 16);

            int* bg_hist_ptr = bg_hist.ptr<int>(b_index);
            bg_hist_ptr[g_index] = bg_hist_ptr[g_index] + 1;
        }
    }
    cv::Mat gray1, gray2, sobel1, sobel2, grad_x1, grad_y1, grad_x2, grad_y2;
    //converting into grayscale
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);

    // Sobel filter for gradient in X direction
    cv::Sobel(gray1, grad_x1, CV_64FC1, 1, 0, 3);
    // Sobel filter for gradient in Y direction
    cv::Sobel(gray1, grad_y1, CV_64FC1, 0, 1, 3);
    cv::convertScaleAbs(grad_x1, grad_x1);
    cv::convertScaleAbs(grad_y1, grad_y1);

    //creating the region of interests
    cv::Rect img_1_r1(grad_x1.cols / 4, grad_x1.rows / 4, grad_x1.cols / 2, grad_x1.rows / 2);
    cv::Rect img_1_r2(grad_x1.cols / 4, grad_x1.rows / 4, grad_y1.cols / 2, grad_y1.rows / 2);
    cv::Mat img_1_roi1 = grad_x1(img_1_r1);
    cv::Mat img_1_roi2 = grad_y1(img_1_r2);
    
    //creating the histograms using magnitude and orientation
    for (int i = 0; i < img_1_roi1.rows; i++) {
        cv::Vec3b* x_ptr1 = img_1_roi1.ptr<cv::Vec3b>(i);
        cv::Vec3b* y_ptr1 = img_1_roi2.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img_1_roi1.cols; j++) {

            int mag_index = int((std::sqrt(x_ptr1[j][0] * x_ptr1[j][0] + y_ptr1[j][0] * y_ptr1[j][0])) / 45.078);
            int orientation_index = int((atan2(y_ptr1[j][0], x_ptr1[j][0]) + 3.14) / 0.785);

            cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(mag_index);
            hist_ptr[orientation_index][0]++;
        }
    }
    
    //creating the feature vector
    for (int i = 0; i < bg_hist.rows; i++) {
        int* bg_hist_ptr = bg_hist.ptr<int>(i);
        for (int j = 0; j < bg_hist.cols; j++) {
            f_vector.push_back(bg_hist_ptr[j]);
        }
    }
    f_vector.push_back(0.1);        //pushing something random to distinguish the histograms
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* hist_ptr = hist.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hist.cols; j++) {
            f_vector.push_back(hist_ptr[j][0]);
        }
    }
    return f_vector;
}
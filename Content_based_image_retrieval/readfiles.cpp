/*
  Bruce A. Maxwell
  S21

  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include"csv_util.h"

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char* argv[]) {
    char dirname[256];
    char baseline_file[256];
    char histogram_file[256];
    char multi_histogram_file[256];
    char TC_file[256];
    char custom_design_file[256];
    char extension_file[256];
    char buffer[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    int i;

    // check for sufficient arguments
    if (argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // get the directory path
    strcpy(dirname, argv[1]);
    strcpy(baseline_file, "baseline_file.csv");
    strcpy(histogram_file, "histogram_file.csv");
    strcpy(multi_histogram_file, "multi_histogram_file.csv");
    strcpy(TC_file, "TC_file.csv");
    strcpy(extension_file, "extension_file.csv");
    strcpy(custom_design_file, "custom_design_file.csv");

    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            printf("full path name: %s\n", buffer);

            std::vector<float> baseline_f_vector = baseline_feature(buffer);
            std::vector<float> histogram_f_vector = histogram_feature(buffer);
            std::vector<float> multi_histogram_f_vector = multi_histogram_feature(buffer);
            std::vector<float> TC_f_vector = TC_feature_1(buffer);
            std::vector<float> custom_design_f_vector = custom_design_feature_1(buffer);
            std::vector<float> extension_f_vector = extension(buffer);


            append_image_data_csv(baseline_file, buffer, baseline_f_vector);
            append_image_data_csv(histogram_file, buffer, histogram_f_vector);
            append_image_data_csv(multi_histogram_file, buffer, multi_histogram_f_vector);
            append_image_data_csv(TC_file, buffer, TC_f_vector);
            append_image_data_csv(custom_design_file, buffer, custom_design_f_vector);
            append_image_data_csv(extension_file, buffer, extension_f_vector);
        }
    }

    printf("Terminating\n");

    return 0;
}



/*
Author - Priyen Shah

Employee ID - 141541

  - Write a program to enhance the image visibility using linear contrast stretching
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;


void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

//This function calculates the histogram vector and saves it to the object received via argument.
void calcHist(cv::Mat& mat, std::vector<int>& hist, bool verbose = false){
    // initialize all intensity values to 0
    for(int i = 0; i < 256; i++) {
        hist.at(i) = 0;
    }

    for(int row=0;row<mat.rows;row++) {
        for(int col = 0; col < mat.cols; col++) {
            hist.at((int)mat.at<uint8_t>(row,col)) += 1;
        }
    }

    if(verbose){
        std::cout << "histogram: " << std::endl;
        
        for (auto i = hist.begin(); i != hist.end(); ++i) 
        std::cout << *i << " "; 
    }

}

//This function accesses the histogram vector through argument along with the title for the plot.
void plotHist(std::vector<int> &hist, std::string title){
    
    plt::plot(hist);
    plt::title(title);
    plt::xlabel("Pixel Intensity");
    plt::ylabel("Number of Pixels");
    plt::show();

}

cv::Mat stretchImage(cv::Mat& mat) {
    // allcoate memory for no of pixels for each intensity value
    std::vector<int>hist(256);
    bool verbose = false;
    cv::Mat contrast = cv::Mat::zeros(mat.rows,mat.cols, CV_8U);

    int minIntensity = 0;
    int maxIntensity = 255;

    calcHist(mat, hist);

    //Maximum intensity in image
    int highest_p;

     for (int i = (hist.size()-1); i >= 0; --i) {
        if (hist.at(i) != 0) {
            highest_p=i;
            break;
        }
    }

    //Minimum intensity in image
    int lowest_p;
    int sum = 0;
     for (int i = 0; i < hist.size(); ++i) {   
        if (hist.at(i) != 0) {
            lowest_p = i; 
            break;
         }
    }

    //Linear Contrast Stretching of Image
    for(int row=0;row<mat.rows;row++) {
        for(int col = 0; col < mat.cols; col++) {
            contrast.at<uint8_t>(row, col) =  (mat.at<uint8_t>(row, col) - lowest_p) * ((maxIntensity - minIntensity)/(highest_p - lowest_p));
        }
    }

    return contrast;
}


int main() {
    //Input the image
    cv::Mat image = cv::imread("contrastTest.jpg" , 0);
    cv::Mat stretchedImage = cv::Mat::zeros(image.rows,image.cols, CV_8U);

    //Show input Image
    cv::imshow( "input Image", image);

    //Perform Histogram plotting  
    stretchedImage = stretchImage(image);

    //Show input Image
    cv::imshow( "Linear contrast stretched Image", stretchedImage);

    //Save Image
    saveImageToFile(stretchedImage, "linearContrastOutput.jpg");
    
    //wait and quit  
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

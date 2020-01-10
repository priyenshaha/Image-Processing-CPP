/*

Author: Priyen Shah - 141541

Write a program to plot a histogram

Acceptance Criteria-
 
 - Each program functionality should be written in a separate function and should be written in Python- numpy  / C++

*/

#include <opencv2/opencv.hpp>
#include "matplotlibcpp.h"
#include <stdint.h>
#include <iostream>
//Vectors are required to save the pixel intensity values.
//We use vectors as the matplotlibcpp needs vectors to plot
#include <vector>

//String is used to print the title in plot
#include <string>

namespace plt = matplotlibcpp;

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

int main(){

	cv::Mat image = cv::imread("histEqTest.jpeg", 0);
	cv::imshow("Input Image", image);

	std::vector<int> hist(256);
	
	calcHist(image, hist);
	plotHist(hist, "Input image histogram");

	return 0;
}
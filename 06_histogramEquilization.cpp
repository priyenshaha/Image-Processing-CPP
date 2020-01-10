#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <string>

#include "matplotlibcpp.h"

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

int histEq(cv::Mat& mat) {
    // allcoate memory for no of pixels for each intensity value
    std::vector<int>histogram(256);
    calcHist(mat, histogram);
    
    std::vector<double>probability(256);
    std::vector<double>cum_sum(256);
    

    //Probability = Pixel at that intensity/Total number of pixels
    for(int i = 0; i < 256; i++) {
        probability.at(i) = (double)histogram.at(i)/(mat.rows*mat.cols);
    }

    //Taking Cumulative Sum of Probability
    std::partial_sum(probability.begin(), probability.end(), cum_sum.begin());

    //Multiply by (2^n-1), here n is number of bits
    for(int i = 0; i < 256; i++) {
        cum_sum.at(i) *= 255;
    }

    //Taking Floor Rounding of cumulative sum
    for(int i = 0; i < 256; i++) {
        cum_sum.at(i) = round(cum_sum.at(i));
    }

    cv::Mat contrast_img = cv::Mat::zeros(mat.rows,mat.cols, CV_8U);

    //Converting cum_sum to new Image form
    for(int row=0;row<mat.rows;row++) {
        for(int col = 0; col < mat.cols; col++) {
            contrast_img.at<uint8_t>(row, col) = cum_sum.at(mat.at<uint8_t>(row, col));
        }
    }
    contrast_img.copyTo(mat);

    
    //Plotting Histogram of input Image
    plotHist(histogram, "Input image histogram");

    calcHist(mat, histogram);

    plotHist(histogram, "Enhanced image histogram");

 }

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}


int main() {
    //Loads image
    cv::Mat image = cv::imread("histEqTest.jpeg" , 0);

    //Show input Image
    cv::imshow( "Input Image", image );

    //Perform Histogram Equaliztion   
    histEq(image);

    //Display Histogram equalized image
    cv::imshow( "Histogram Enhanced image", image);

    // Save to file
    saveImageToFile(image, "histogramEquilization.jpg");

    //wait and quit  
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
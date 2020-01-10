/*
Author - Priyen Shah
Emp ID - 141541
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

void GaussianFilterCreation(float GKernel[][5]) 
{ 
    // intialising standard deviation to 1.0 
    float sigma = 1.0; 
    float r, s = 2.0 * sigma * sigma; 
  
    // sum is for normalization 
    float sum = 0.0; 
  
    // generating 5x5 kernel 
    for (int x = -2; x <= 2; x++) { 
        for (int y = -2; y <= 2; y++) { 
            r = sqrt(x * x + y * y); 
            GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s); 
            sum += GKernel[x + 2][y + 2]; 
        } 
    } 
  
    // normalising the Kernel 
    for (int i = 0; i < 5; ++i) 
        for (int j = 0; j < 5; ++j) 
            GKernel[i][j] /= sum; 
} 

//This function pads and convolves the inputImage with the kernel supplied and stores in the outputImage.
int convolve(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
    //Create a mat object of all zeros. The data type should be same as input image. Here unsigned int of 8 bits
    cv::Mat paddedImage = cv::Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_8U);

    //Padding of image
    //Add the input image in to the paddedImage mat obj. Leave the size of padding aside. So initialize the row and col counter accordingly
    for(int row=(int)kernel.rows/2;row<paddedImage.rows-(int)kernel.rows/2;row++) {
        for(int column=(int)kernel.cols/2;column<paddedImage.cols-(int)kernel.cols/2;column++) {
            // Ensure that the input image data is taken from zeroth index.
            paddedImage.at<uint8_t>(row, column) = inputImage.at<uint8_t>(row-(int)kernel.rows/2, column-(int)kernel.cols/2);
        }
    }

    //Convolve the image with the input filter

    for(int row=(int)kernel.rows/2;row<((paddedImage.rows)-(int)kernel.rows/2);row++) {
        for(int column=(int)kernel.cols/2;column<((paddedImage.cols)-(int)kernel.cols/2);column++) {
            int t_sum = 0;
            for(int k_row=-(int)kernel.rows/2;k_row<(kernel.rows-(int)kernel.rows/2);k_row++) {
                for(int k_column=-(int)kernel.cols/2;k_column<(kernel.cols-(int)kernel.cols/2);k_column++) {
                    t_sum += (int)((int)paddedImage.at<uint8_t>(row+k_row, column+k_column) * (float)kernel.at<float>(k_row+(int)kernel.rows/2, k_column+(int)kernel.cols/2));
                }
            }
            outputImage.at<uint8_t>(row-(int)kernel.rows/2, column-(int)kernel.cols/2) = t_sum;
        }
    }  
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat inputImage = cv::imread("fish.png",0);
    cv::imshow("Input image", inputImage);
    std::cout<<"input img dim: " << inputImage.size<<std::endl;

    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8U);	

    // Create an array for kernel and pass it to create the gaussian filter.

    float kernelData[5][5];
    GaussianFilterCreation(kernelData);

    //Create mat obj of the kernel.
    cv::Mat gaussianKernel(5,5, CV_32F, kernelData);

    convolve(inputImage, outputImage, gaussianKernel);

    cv::imshow("blurred",outputImage);
	
    saveImageToFile(outputImage, "gaussianBlurred.jpg");

    cv::waitKey(0);

	return 0;
}
/*
Author: Priyen Shah - 141541

Assignment 1, Code A

Description :
 - Write a program to open image data and render it. Use openCV for this. Debug the program to
i)  Identify the variable (array / Pointer) which indicates the start of the image data. 
ii) Identify the meta information about the image viz. image width, image height, no. of bytes per pixel,  format of image data etc

Acceptance Creiteria:
- The functionality should be written in a separate function and should be written in Python- numpy  / C++
DoD : 
- Understanding of nature  of image data and accessing it for various operations pixel by pixel (1x1 pixel) OR in pixel block  (nxn)

*/

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include<stdint.h>

// This function returns the type of image
std::string getNBytes(int idx){

	// CV_8U breaks down as: 8 unsigned bits for 1 channel
	// CV_16UC3 breaks down as: 16 unsigned bits per channel. 3 channels.
	std::string byteInfoArray[6] = {"8 bits / 1 byte, unsigned", "8 bits / 1 byte, signed", "16 bits / 2 bytes, unsigned", "16 bits / 2 bytes, signed", "32 bits / 4 bytes, Float", "64 bits / 8 bytes,Float"};

	return byteInfoArray[idx];
}

void showImageMetaData(cv::Mat imgObj){

	int i_height = imgObj.rows;
	int i_width = imgObj.cols;
	int nChannels = imgObj.channels();
	int depth = imgObj.depth();

	//unsigned char* imageData = (unsigned char*)(image.data);

	std::cout << "Height of Image: " << i_height << std::endl;
	std::cout << "Width of Image: " << i_width << std::endl;
	std::cout << "Channels in Image: " << nChannels << std::endl;
	std::cout << "Number of bytes per pixel: " << getNBytes(depth) << std::endl; //CV_8U = 0

}

int globalAverage(cv::Mat image){

	int Totalintensity = 0;
	for (int i=0; i < image.rows; ++i){
	    for (int j=0; j < image.cols; ++j){
	        Totalintensity += (int)image.at<uint8_t>(i, j);
	    }
	}

	// Find avg lum of frame
	float avgLum = 0;
	return(Totalintensity/(image.rows * image.cols));


}

int main(){

	// Creates a Mat object to contain image headers and data
	cv::Mat image;

	// Reads the image from given file. 2nd arg - 0 reads the image into grayscale
	image=cv::imread("fish.png", 0);

	// Show the image
	cv::imshow("Original Image", image);

	// This function shows the Meta data of the image
	showImageMetaData(image);
	
	// at function template is used to access the pixel at a specific location
	std::cout << "Pixel Value at [120, 120]: " << (int)image.at<uint8_t>(120,120) << std::endl;

	// This function gets the global average value 
	std::cout << "Average Intensity: " << globalAverage(image) << std::endl;

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;

}
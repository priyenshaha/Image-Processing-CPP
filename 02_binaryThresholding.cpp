/*
Author: Priyen Shah - 141541

Write a program for binary thresholding

Modify the above program for adaptive thresholding. In this Threshold value should be calculated depending on content of the image data

Acceptance Criteria:
- Store the Output image as another image file
- The functionality should be written in a separate function and should be written in Python- numpy  / C++

DoD : 
- Understanding of nature  of image data and accessing it for various operations pixel by pixel (1x1 pixel) OR in pixel block  (nxn)\
    

*/

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include<stdint.h>


// This function returns the global average intensity of all the pixels.
int globalAverage(cv::Mat image){

	//iterate over the entire image pixel by pixel
	int Totalintensity = 0;
	for (int h=0; h < image.rows; ++h){
	    for (int w=0; w < image.cols; ++w){
	        Totalintensity += (int)image.at<uint8_t>(h, w);
	    }
	}

	// Find avg intensity of frame
	float avgLum = 0;
	return(Totalintensity/(image.rows * image.cols));


}

// This function returns the mat object of the binary thresholded image (default threshold = 127)
cv::Mat binaryThresholding(cv:: Mat image, int thresh=127){

	//iterate over the entire image pixel by pixel
	for (int h=0; h < image.rows; ++h){
	    for (int w=0; w < image.cols; ++w){
	        if((int)image.at<uint8_t>(h, w)>thresh)
	        	image.at<uint8_t>(h, w) = 255;

	        else
	        	image.at<uint8_t>(h, w)=0;
	    }
	}

	return image;
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}


int main(){

	cv::Mat image = cv::imread("fish.png", 0);
	cv::Mat binImg;

	cv::imshow("Grayscale Image", image);
	binImg = binaryThresholding(image, globalAverage(image));

	cv::imshow("binary Image", binImg);


	saveImageToFile(binImg, "BinaryThresholdedImage.jpg");

	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}
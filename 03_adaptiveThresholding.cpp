/*
Author: Priyen Shah - 141541

Created on Tue Oct 22 20:35:47 2019

Write A program for adaptive thresholding 

Acceptance criteria:
 - Store the Output image as another image file
 - The functionality should be written in a separate function and should be written in Python- numpy  / C++
*/


#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<stdint.h>

using namespace std;

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

void adaptiveThresholding(cv::Mat src){

	int width = src.cols;
	int height = src.rows;
	int GRID_SIZE=80;
	
	for (int y = 0; y <= height - GRID_SIZE; y += GRID_SIZE) {
	    for (int x = 0; x <= width - GRID_SIZE; x += GRID_SIZE) {

	    	cv::Mat window = src(cv::Rect(x, y, GRID_SIZE, GRID_SIZE));
	    	window = binaryThresholding(window, globalAverage(window));
	        window.copyTo(src(cv::Rect(x, y, GRID_SIZE, GRID_SIZE)));

	    }
	}
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat image = cv::imread("sudoku.jpg", 0);
	
	cv::imshow("Grayscale image", image);
	cout<<" img dim: " << image.size() << endl; 

	adaptiveThresholding(image);
	cv::imshow("Adap image", image);

	saveImageToFile(image, "adaptiveThresholdedImage.jpg");
	
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
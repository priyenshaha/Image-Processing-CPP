/*

*	Author - Priyen Shah
*	Emp ID - 141541

*/

#include <iostream>
#include <string>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cmath>

int interpolate(const cv::Mat& inputImage, float newX, float newY){

	//nearby pixels locations

    int y1 = floor(newY);
    int x1 = floor(newX);
    int y2 = y1+1;
    int x2 = x1+1;

    // nearby pixel values
	int p11 = (int)inputImage.at<uint8_t>(x1, y1);
	int p12 = (int)inputImage.at<uint8_t>(x1, y2);
	int p21 = (int)inputImage.at<uint8_t>(x2, y1);
	int p22 = (int)inputImage.at<uint8_t>(x2, y2);
	
	//interpolation values
	int output=0, bottom=0, top=0;

	if(x2 < inputImage.rows && y2 < inputImage.cols){

		bottom = p11 * (x2-newX)/(x2-x1)		//bottom left
			   + p21 * (newX-x1)/(x2-x1);		//top left

		top = p12 * (x2-newX)/(x2-x1)			//bottom right
			+ p22 * (newX-x1)/(x2-x1);			//top right

		output = bottom * ((y2-newY)/(y2-y1)) + top * ((newY-y1)/(y2-y1));
	}
	
	return output;
}

void scale(cv::Mat& inputImage, cv::Mat& scaledImage, float scaleX, float scaleY){
	//Finding the pixels surrounding the expected pixel location post scaling
	//Let x and y be new coordinates post scaling. Find pixels in input image that are nearby the expected location
	int centerH = (int)(inputImage.rows / 2);
	int centerW = (int)(inputImage.cols / 2);

	// Set the centre of inputImage as origin for rotation.
	float scaledMatrixData []= { scaleX,	0,		centerH, 
						 		   0, 		scaleY, centerW, 
						  		   0, 		0, 		1		 };

	cv::Mat scaledMatrix(3, 3, CV_32F, scaledMatrixData);

	for(int row = 0; row < inputImage.rows; row++){
		for(int col = 0; col < inputImage.cols; col++){

			//shift the pixels according to the new origin
			float vectData []= { (row - centerH),
								 (col - centerW), 
								 1				 };

			cv::Mat vect(3, 1, CV_32F, vectData);

			cv::Mat transformedVect = scaledMatrix * vect;

			if(	   transformedVect.at<float>(0,0)<inputImage.rows
				&& transformedVect.at<float>(1,0)<inputImage.cols	//

				&& transformedVect.at<float>(0,0)>=0
				&& transformedVect.at<float>(1,0)>=0){

				scaledImage.at<uint8_t>(row,col) = interpolate(inputImage, transformedVect.at<float>(0,0), transformedVect.at<float>(1,0));

			}
		}
	}
}

int main(){

	cv::Mat inputImage = cv::imread("/home/kpit/assignment/images/affineInput.png", 0);

	float scaleX = 0.5f, scaleY = 0.5f;

	//Create the output image of dimensions considering the scaling factors.
	cv::Mat scaledImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);

	scale(inputImage, scaledImage, scaleX, scaleY);

	cv::imshow("Input image", inputImage);
	cv::imshow("Scaled image", scaledImage);

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}


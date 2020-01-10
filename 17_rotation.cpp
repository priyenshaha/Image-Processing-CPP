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


double pi(){ 
    return std::atan(1)*4; 
}

cv::Mat rotate(const cv::Mat& inputImage, const int degree){

    cv::Mat rotatedImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);

	float radian = degree * pi()/180;
	float centerY = (int)(inputImage.rows / 2);
	float centerX = (int)(inputImage.cols / 2);

	// Set the centre of inputImage as origin for rotation.
	float rotationMatrixData []= { cos(radian), sin(radian), centerH, 
						 		  -sin(radian), cos(radian), centerW, 
						  		   0, 		    0, 			 1		 };

	cv::Mat rotationMatrix(3, 3, CV_32F, rotationMatrixData);

	for(int row = 0; row < inputImage.rows; row++){
		for(int col = 0; col < inputImage.cols; col++){

			//shift the pixels according to the new origin
			float vectData []= { (row - centerY),
								 (col - centerX), 
								 1				 };

			cv::Mat vect(3, 1 , CV_32F, vectData);

			cv::Mat transformedVect = rotationMatrix * vect;

			if(	   transformedVect.at<float>(0,0)<inputImage.rows
				&& transformedVect.at<float>(1,0)<inputImage.cols

				&& transformedVect.at<float>(0,0)>=0
				&& transformedVect.at<float>(1,0)>=0){

				rotatedImage.at<uint8_t>(row,col) = interpolate(inputImage, transformedVect.at<float>(0,0), transformedVect.at<float>(1,0));

			}			
		}
	}

	return rotatedImage;

}

int main(){

	cv::Mat inputImage = cv::imread("affineInput.png", 0);
	cv::Mat rotatedImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);

	int angle;
	std::cout << "Enter Angle in degrees to rotate: ";
	std::cin >> angle;

	cv::imshow("Input image", inputImage);
	cv::imshow("Rotated image", rotate(inputImage,angle));

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
/*

*	Author - Priyen Shah
*	Emp ID - 141541

*/

#include <iostream>
#include <string>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cmath>

void AffineTransform(const cv::Mat& src, cv::Mat& dst, cv::Mat& translateTransform){
	// Check entered translation points are within image bounds
	if((abs((int)translateTransform.at<double>(0,2))) > src.rows || (abs((int)translateTransform.at<double>(1,2))) > src.cols){

		std::cout<<"Initialised Translate points exceeds image bounds"<<std::endl;
		
	}
	// Create an empty 3x1 matrix for storing original frame coordinates
	cv::Mat xOrg = cv::Mat(3, 1, CV_64FC1);

	// Create an empty 3x1 matrix for storing transformed frame coordinates
	cv::Mat xTrans = cv::Mat(3, 1, CV_64FC1); 

	// Default initialisation of output matrix
	dst = cv::Mat::zeros(src.rows, src.cols, src.type());

	// Go through entire image
	for(int row = 0; row < src.size().height; row++){
		for(int col = 0; col < src.size().width; col++){
		// Get current coorndinates
			xOrg.at<double>(0,0) = row;
			xOrg.at<double>(1,0) = col;
			xOrg.at<double>(2,0) = 1;
		
		// Get transformed coodinates

			xTrans = translateTransform * xOrg;
			
			// Depth
			const int w = (xTrans.at<double>(2,0));
			// Homogeneous to cartesian transformation
			const int newX = (xTrans.at<double>(0,0)) / w;
			const int newY = (xTrans.at<double>(1,0)) / w;

			//Make sure boundary is not exceeded
			if(newX >= src.size().height || newY >= src.size().width || newX < 0 || newY < 0){
				continue;
			}
			
			// Put the values of original coordinates to transformed coordinates
			dst.at<uchar>(newX, newY) = src.at<uchar>(row, col);
			
		}
	}

}

double pi(){ 
    return std::atan(1)*4; 
}

int main(){

	cv::Mat inputImage = cv::imread("affineInput.png", 0);
	cv::Mat translatedImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
	int translateW = 0, translateH = 0;

	std::cout << "Enter Height to translate: ";
	std::cin >> translateH;

	std::cout << "Enter width to translate: ";
	std::cin >> translateW;

	cv::Mat translateTransform(3,3,CV_64FC1);
	// Initialise translational, rotational and scaling values for transformation
	translateTransform.at<double>(0,0) = 1;
	translateTransform.at<double>(0,1) = 0;
	translateTransform.at<double>(0,2) = translateH;

	translateTransform.at<double>(1,0) = 0;
	translateTransform.at<double>(1,1) = 1;
	translateTransform.at<double>(1,2) = translateW;

	translateTransform.at<double>(2,0) = 0;
	translateTransform.at<double>(2,1) = 0;
	translateTransform.at<double>(2,2) = 1;

	std::cout << "Translation Transform Mat: " << std::endl;

	for(int x=0; x < translateTransform.rows; x++){
		for(int y=0; y < translateTransform.cols; y++){
			std::cout << (double)translateTransform.at<double>(x,y) << "\t";
		}
		std::cout << ";" << std::endl;
	}

	AffineTransform(inputImage, translatedImage, translateTransform);

	cv::imshow("Input image", inputImage);
	cv::imshow("Translated image", translatedImage);

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
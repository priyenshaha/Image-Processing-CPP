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

		output = (int)(bottom * ((y2-newY)/(y2-y1)) + top * ((newY-y1)/(y2-y1)));
	}
	
	return output;
}


bool AffineTransform(const cv::Mat& inputImage, cv::Mat& outputImage, const cv::Mat& affineTransform, float totalShiftX, float totalShiftY){

	// Go through entire image
	for(int row = 0; row < outputImage.rows; row++){
		for(int col = 0; col < outputImage.cols; col++){

			//Select the pixels in output image with the required shift
			float vectData []= { (row - totalShiftY),
								 (col - totalShiftX), 
								 1				 };

			cv::Mat vect(3, 1 , CV_32F, vectData);
		
			// Get transformed coodinates
			cv::Mat xTrans = affineTransform * vect;

			const int newX = (xTrans.at<float>(0,0));
			const int newY = (xTrans.at<float>(1,0));

			//Consider the transformed pixels only if they lie inside the outputImage dimensions
			if( (newX < inputImage.rows) && (newY < inputImage.cols) &&	(newX >= 0) && (newY >= 0)){
	
				// Put the values of original coordinates to transformed coordinates
				outputImage.at<uchar>(row, col) = interpolate(inputImage, newX, newY);
			}
			
		}
	}

	return 1;

}

float pi(){ 
    return std::atan(1)*4; 
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat inputImage = cv::imread("affineInput.png", 0);
	cv::imshow("Input image", inputImage);

	float theta = 45, scaleX = 1.2, scaleY = 1.2, translateX = 100, translateY = 5;
	
	//Center of inputImage. This is used as origin for rotation.
	float centerX = inputImage.cols/2;
	float centerY = inputImage.rows/2;

	//Shift required for translation
	float transShiftX = translateX/2;
	float transShiftY = translateY/2;

	//Dimension of the resultant image after affine transformation
	int newWidth = (inputImage.rows/scaleY)*sin(theta * pi()/180) + (inputImage.cols/scaleX)*cos(theta * pi()/180) + abs(translateX);
	int newHeight = (inputImage.rows/scaleY)*cos(theta * pi()/180) + (inputImage.cols/scaleX)*sin(theta * pi()/180)+ abs(translateY);

	//Additional shifting of pixels required to fit the transformed image in the window
	float imageShiftX = (newWidth - inputImage.cols)/2 ;
	float imageShiftY = (newHeight - inputImage.rows)/2;

	//Calculate the total shift to map the pixels to resultant image
	float totalShiftX = centerX - transShiftX + imageShiftX;
	float totalShiftY = centerY - transShiftY + imageShiftY;

	float radian = theta * pi()/180;

	//Matrix for translation
	float translationMatrixData []= { 1, 0, -translateY, 
						 		  	  0, 1, -translateX, 
						  		   	  0, 0, 1			};

	cv::Mat translationMatrix(3, 3, CV_32F, translationMatrixData);

	//Matrix for rotation
	float rotationMatrixData []= { cos(radian), sin(radian), centerY, 
						 		  -sin(radian), cos(radian), centerX, 
						  		   0, 		    0, 			 1		 };
					  		   
	cv::Mat rotationMatrix(3, 3, CV_32F, rotationMatrixData);

	//Matrix for scaling
	float scalingMatrixData []= { scaleY, 0, 	  0, 
						 		  0, 	  scaleX, 0, 
						  		  0, 	  0, 	  1  };

	cv::Mat scalingMatrix(3, 3, CV_32F, scalingMatrixData);

	//All these matrices multiplied together gives a resultant matrix which is our affine transform matrix
	cv::Mat affineTransform = (rotationMatrix * scalingMatrix) * translationMatrix ;

	//Printing the resultant transform matrix
	std::cout << "Affine Transform Mat: " << std::endl;

	for(int x=0; x < affineTransform.rows; x++){
		for(int y=0; y < affineTransform.cols; y++){
			std::cout << (float)affineTransform.at<float>(x,y) << "\t";
		}
		std::cout << ";" << std::endl;
	}

	cv::Mat translatedImage = cv::Mat::zeros(newHeight, newWidth, CV_8U);

	AffineTransform(inputImage, translatedImage, affineTransform, totalShiftX, totalShiftY);
	cv::imshow("Translated image", translatedImage);

	saveImageToFile(translatedImage, "affineTransformOutputImage.jpg")

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
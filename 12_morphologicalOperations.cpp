/*

*	Author - Priyen Shah
*	Emp ID - 141541

*	Problem Statement:
	
	Write a program to apply the Morphological (Dilate, Erosion, Open , Close) operations  for image filtering. 
	
	i) Define 3x3 element kernel for morphological operations 

*/

#include <opencv2/opencv.hpp>
#include <string>
#include <stdint.h>
#include <iostream>

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
void binaryThresholding(cv:: Mat& image, int thresh=127){

	//iterate over the entire image pixel by pixel
	for (int h=0; h < image.rows; ++h){
	    for (int w=0; w < image.cols; ++w){
	        if((int)image.at<uint8_t>(h, w)>thresh)
	        	image.at<uint8_t>(h, w) = 255;

	        else
	        	image.at<uint8_t>(h, w)=0;
	    }
	}
}

// This function dilates the image like dilating with a 3x3 diamond kernel
void dilate(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat kernel){
	
	int intensity=255;
	
	for (int row = 0; row < inputImage.rows; row++){
		for (int col = 0; col < inputImage.cols; col++){
			
			if (inputImage.at <uint8_t>(row, col) == intensity){

				outputImage.at<uint8_t>(row, col) = intensity;

				if (row > 0)
					outputImage.at<uint8_t>(row - 1, col) = intensity;

				if (col > 0)
					outputImage.at<uint8_t>(row, col - 1) = intensity;

				if ((row + 1) < inputImage.rows)
					outputImage.at<uint8_t>(row + 1, col) = intensity;

				if ((col + 1) < inputImage.cols)
					outputImage.at<uint8_t>(row, col + 1) = intensity;
			}
		}
	}
	
}

// This function erodes the image like eroding with a 3x3 diamond kernel
void erode(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat kernel){
	
	int intensity=0;
	
	for (int row = 0; row < inputImage.rows; row++){
		for (int col = 0; col < inputImage.cols; col++){
			
			if (inputImage.at <uint8_t>(row, col) == intensity){

				outputImage.at<uint8_t>(row, col) = intensity;
				if (row > 0)
					outputImage.at<uint8_t>(row - 1, col) = intensity;
				if (col > 0)
					outputImage.at<uint8_t>(row, col - 1) = intensity;
				if ((row + 1) < inputImage.rows)
					outputImage.at<uint8_t>(row + 1, col) = intensity;
				if ((col + 1) < inputImage.cols)
					outputImage.at<uint8_t>(row, col + 1) = intensity;
			}
		}
	}
	
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){
	
	cv::Mat inputImage = cv::imread("morphInputImage.png", 0);
	// Convert the grayscale image into binary image.
	binaryThresholding(inputImage, globalAverage(inputImage));

	cv::imshow("Input binary Image", inputImage);

	float structuringElementData[] = { 0, 1, 0,
									   1, 1, 1,
									   0, 1, 0 };

	cv::Mat structuringElement(3, 3, CV_32F, structuringElementData);

	cv::Mat dilatedImage = inputImage.clone();
	cv::Mat erodedImage = inputImage.clone();
	
	dilate(inputImage, dilatedImage, structuringElement);
	cv::imshow("Dilated image", dilatedImage);

	erode(inputImage, erodedImage, structuringElement);
	cv::imshow("Eroded image", erodedImage);

	cv::Mat erodedClone = erodedImage.clone();
	cv::Mat openedImage = erodedImage.clone();

	//Opening = erosion followed by dilation
	dilate(erodedClone, openedImage, structuringElement);
	cv::imshow("Opened Image", openedImage);

	cv::Mat dilatedClone = dilatedImage.clone();
	cv::Mat closedImage = dilatedImage.clone();

	//Closing = dilation followed by erosion
	erode(dilatedClone, closedImage, structuringElement);
	cv::imshow("Closed Image", closedImage);

	saveImageToFile(dilatedImage, "dilatedOutputImage.jpg");
	saveImageToFile(erodedImage, "erodedOutputImage.jpg");
	saveImageToFile(openedImage, "openedOutputImage.jpg");
	saveImageToFile(closedImage, "closedOutputImage.jpg");

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
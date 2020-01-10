/*

*	Author - Priyen Shah
*	Emp ID - 141541

*/

#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <iostream>
#include <string>
#include <cmath>

int calcSigma(cv::Mat& kernel){

	int mean=0, sigma=0, idx=0;
	int Ksize = (kernel.rows*kernel.cols);
	int squaredDiffArr[Ksize];

	for(int row=0; row < kernel.rows; row++){
		for(int col=0; col < kernel.cols; col++){

			//std::cout << "Pixel val: " << (int)kernel.at<uint8_t>(row, col) << std::endl;
			mean += (int)kernel.at<uint8_t>(row, col);
		}
	}
	mean = round(mean/Ksize);
	
	//std::cout << "Ksize: " << Ksize <<", mean of window: " << mean << std::endl;                    
	
	for(int row=0; row < kernel.rows; row++){
		for(int col=0; col < kernel.cols; col++){

			sigma += (int)(pow((int)kernel.at<uint8_t>(row, col)-mean, 2));
			
		}
	}
	sigma = round(sigma/Ksize);
	return sqrt(sigma);
}

int sigmaFilter(cv::Mat& kernel){

	int sigma = 0, currentPixel=0, lowerLimit=0, upperLimit=0, cnt=0, sum=0, sigmaMean=0;
	sigma = calcSigma(kernel);
    //std::cout << "Sigma: " << sigma << std::endl;

    for(int row=0; row < kernel.rows; row++){
		for(int col=0; col < kernel.cols; col++){

			currentPixel = (int)kernel.at<uint8_t>(row, col);

			lowerLimit = currentPixel - 2*sigma;
			upperLimit = currentPixel + 2*sigma;

			if(currentPixel >= lowerLimit && currentPixel <= upperLimit){
				++cnt;
				sum += currentPixel;
			}
		}
	}

	sigmaMean = round(sum/cnt);
	return sigmaMean;
            
}

void window(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel){
	
	for(int row = (int)kernel.rows/2; row < ((inputImage.rows) - (int)kernel.rows/2); row++) {
        for(int column = (int)kernel.cols/2; column < ((inputImage.cols)-(int)kernel.cols/2); column++) {
            
            for(int k_row = -(int)kernel.rows/2; k_row<(kernel.rows - (int)kernel.rows/2); k_row++) {
                for(int k_column = -(int)kernel.cols/2; k_column < (kernel.cols - (int)kernel.cols/2); k_column++) {
                    
                 kernel.at<uint8_t>(k_row + (int)kernel.rows/2, k_column + (int)kernel.cols/2) = (int)inputImage.at<uint8_t>(row+k_row, column+k_column);
	                
                }
            }
            
            
            //cv::waitKey(0);
            outputImage.at<uint8_t>(row, column) = sigmaFilter(kernel);
        }
    }

}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat inputImage = cv::imread("sigmaFilterInputImage.jpg",0);
	cv::imshow("Input Image", inputImage);

	cv::Mat kernel = cv::Mat::zeros(3,3,CV_8U);
	cv::Mat outputImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
	window(inputImage, outputImage, kernel);

	cv::imshow("Sigma filtered output", outputImage);

	saveImageToFile(outputImage, "sigmaFilterOutputImage.jpg");

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;

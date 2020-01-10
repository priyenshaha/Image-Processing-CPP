/*

*	Author - Priyen Shah
*	Emp ID - 141541

*	Problem Statement:
	
	Write a program to perform NxN block operation to perform median filtering 
	
	(hint : It is a salt-pepper noise removal technique for image data. 
	Pixles coming under NxN are sorted and median among them is taken as output pixel)

*/

#include<iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <string>
 
//sort the window using insertion sort
//insertion sort is best for this sorting
void insertionSort(int window[])
{
    int temp, i , j;
    for(i = 0; i < 9; i++){
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--){
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}

void medianFilter(cv::Mat& inputImage, cv::Mat& outputImage){

	//create a sliding window of size 9
	int window[9];

    for(int row = 1; row < inputImage.rows - 1; row++){
        for(int col = 1; col < inputImage.cols - 1; col++){

            // Pick up window element

            window[0] = inputImage.at<uchar>(row - 1 ,col - 1);
            window[1] = inputImage.at<uchar>(row, col - 1);
            window[2] = inputImage.at<uchar>(row + 1, col - 1);
            window[3] = inputImage.at<uchar>(row - 1, col);
            window[4] = inputImage.at<uchar>(row, col);
            window[5] = inputImage.at<uchar>(row + 1, col);
            window[6] = inputImage.at<uchar>(row - 1, col + 1);
            window[7] = inputImage.at<uchar>(row, col + 1);
            window[8] = inputImage.at<uchar>(row + 1, col + 1);

            // sort the window to find median
            insertionSort(window);

            // assign the median to centered element of the matrix
            outputImage.at<uchar>(row,col) = window[4];
        }
    }
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

 
int main(){

	cv::Mat inputImage = cv::imread("medianFilterInputImage.png", 0);

	if( !inputImage.data ){ 
		
		std::cout << "No Input image data found" << std::endl;
		return -1; 

	}
	cv::imshow("Input Image with salt and pepper noise", inputImage);

	cv::Mat outputImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);

    medianFilter(inputImage, outputImage);

    cv::imshow("Median Blurred Image", outputImage);
    

	cv::waitKey(0);
	cv::destroyAllWindows();

	saveImageToFile(outputImage, "medianFilterOutputImage.jpg");	 
    return 0;
}
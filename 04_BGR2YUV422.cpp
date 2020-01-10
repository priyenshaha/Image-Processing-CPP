/*

Author: Priyen Shah - 141541

Image read / Writing using file operation and openCV, Pixel by pixel Operation and data model conversion

Description :
- Write a program to convert the RGB image data into YUV 422 format and store the Y-component as image

Acceptance Criteria:
The implementaiton should be done by accessing image data pixel by pixel using image array / pointers
Opencv functions should only be used for image reading and  writing . Image pricessing related functions usage SHOULD NOT be done


*/

#include<iostream>
#include<opencv2/opencv.hpp>

#include<stdint.h>

// This function will convert the BGR image into YUV422 image
void cvtBGR2YUV422(cv::Mat image, cv::Mat yComp){

	int B, G, R;
	

	for(int row=0; row < image.rows; row++ ){
		for(int col=0; col < image.cols; col++){

			B = (int)image.at<cv::Vec3b>(row,col)[0];
			G = (int)image.at<cv::Vec3b>(row,col)[1];
			R = (int)image.at<cv::Vec3b>(row,col)[2];
			
			         
            image.at<cv::Vec3b>(row,col)[0] = 0.299*R + 0.587*G + 0.114*B; //Y component
            yComp.at<uint8_t>(row,col) = (int)image.at<cv::Vec3b>(row,col)[0];
            
            if((col % 2)==0){
                image.at<cv::Vec3b>(row,col)[1] = -0.147*R - 0.289*G + 0.436*B + 128;
                image.at<cv::Vec3b>(row,col)[2] = 0.615*R - 0.515*G - 0.100*B + 128;
            }
            
            else{
                image.at<cv::Vec3b>(row,col)[1] = image.at<cv::Vec3b>(row,col-1)[1];
                image.at<cv::Vec3b>(row,col)[2] = image.at<cv::Vec3b>(row,col-1)[2];
            }

			//std::cout << "B: " << B << ", G: " << G << ", R: " << R << std::endl;

		}
	}
}

int main(){

	cv::Mat image = cv::imread("histEqTest.jpeg");
	cv::Mat yComp = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

	cv::imshow("Input image", image);

	cvtBGR2YUV422(image, yComp);
	
	cv::imshow("YUV422 image", image);
	cv::imshow("Y-component", yComp);

	cv::imwrite("../../outputImages/YUV422OutputImage.jpg", image);
	cv::imwrite("../../outputImages/yComponentOfYUV422Image.jpg", yComp);
	
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
/*

*	Author - Priyen Shah
*	Emp ID - 141541

*/

#include <iostream>
#include <string>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

/*
*	Function: convertToGray
	*	InputImage data type- CV_8UC3, outputImage data type - CV_8UC1
	*	Converts BGR image into grayscale image
	
*/
void convertToGray(const cv::Mat& inputImage, cv::Mat& outputImage){

	int B, G, R;

	for(int row=0; row < inputImage.rows; row++ ){
		for(int col=0; col < inputImage.cols; col++){

			B = (int)inputImage.at<cv::Vec3b>(row,col)[0];
			G = (int)inputImage.at<cv::Vec3b>(row,col)[1];
			R = (int)inputImage.at<cv::Vec3b>(row,col)[2];
						         
            outputImage.at<uint8_t>(row, col) = (int)(0.299*R + 0.587*G + 0.114*B); //GrayScale
        }        
	}
}
void GaussianFilterCreation(float GKernel[][5]) 
{ 
    // intialising standard deviation to 1.0 
    float sigma = 1.0; 
    float r, s = 2.0 * sigma * sigma; 
  
    // sum is for normalization 
    float sum = 0.0; 
  
    // generating 5x5 kernel 
    for (int x = -2; x <= 2; x++) { 
        for (int y = -2; y <= 2; y++) { 
            r = sqrt(x * x + y * y); 
            GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s); 
            sum += GKernel[x + 2][y + 2]; 
        } 
    } 
  
    // normalising the Kernel 
    for (int i = 0; i < 5; ++i) 
        for (int j = 0; j < 5; ++j) 
            GKernel[i][j] /= sum; 
}

int convolution_for_32_bit_mat(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
    //Create a inputImage object of all zeros. The data type should be same as input image. Here unsigned int of 32 bits
    cv::Mat paddedImage = cv::Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_32F);

    //Padding of image    
    for(int row=(int)kernel.rows/2;row<paddedImage.rows-(int)kernel.rows/2;row++) {
        for(int col=(int)kernel.cols/2;col<paddedImage.cols-(int)kernel.cols/2;col++) {
            // Ensure that the input image data is taken from zeroth index.
            paddedImage.at<float>(row, col) = inputImage.at<uint8_t>(row-(int)kernel.rows/2, col-(int)kernel.cols/2);
        }
    }

    //Convolution 
    for(int row=(int)kernel.rows/2;row<((paddedImage.rows)-(int)kernel.rows/2);row++) {
        for(int col=(int)kernel.cols/2;col<((paddedImage.cols)-(int)kernel.cols/2);col++) {
            float t_sum = 0;
            for(int k_row=-(int)kernel.rows/2; k_row<(kernel.rows-(int)kernel.rows/2); k_row++) {
                for(int k_col=-(int)kernel.cols/2; k_col<(kernel.cols-(int)kernel.cols/2); k_col++) {
                    t_sum += (float)((float)paddedImage.at<float>(row+k_row, col+k_col) * (float)kernel.at<float>(k_row+(int)kernel.rows/2, k_col+(int)kernel.cols/2));                    
                }
            }
            outputImage.at<float>(row-(int)kernel.rows/2, col-(int)kernel.cols/2) = t_sum;
        }
    } 
}

int convolution(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
    //Create a inputImage object of all zeros. The data type should be same as input image. Here unsigned int of 32 bits
    cv::Mat paddedImage = cv::Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_32F);

    //Padding of image    
    for(int row=(int)kernel.rows/2;row<paddedImage.rows-(int)kernel.rows/2;row++) {
        for(int col=(int)kernel.cols/2;col<paddedImage.cols-(int)kernel.cols/2;col++) {
            // Ensure that the input image data is taken from zeroth index.
            paddedImage.at<float>(row, col) = inputImage.at<float>(row-(int)kernel.rows/2, col-(int)kernel.cols/2);
        }
    }

    //Convolution 
    for(int row=(int)kernel.rows/2;row<((paddedImage.rows)-(int)kernel.rows/2);row++) {
        for(int col=(int)kernel.cols/2;col<((paddedImage.cols)-(int)kernel.cols/2);col++) {
            float t_sum = 0;
            for(int k_row=-(int)kernel.rows/2; k_row<(kernel.rows-(int)kernel.rows/2); k_row++) {
                for(int k_col=-(int)kernel.cols/2; k_col<(kernel.cols-(int)kernel.cols/2); k_col++) {
                    t_sum += (float)((float)paddedImage.at<float>(row+k_row, col+k_col) * (float)kernel.at<float>(k_row+(int)kernel.rows/2, k_col+(int)kernel.cols/2));                    
                }
            }
            outputImage.at<float>(row-(int)kernel.rows/2, col-(int)kernel.cols/2) = t_sum;
        }
    } 
}

/*
*	Function: GaussianBlur
	*	Applies gaussian filter to the inputImage. 
	*	InputImage and outputImage data type: CV_32F
*/
cv::Mat gaussianBlur(const cv::Mat& inputImage){

	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_32F);

    float gaussianKernelData[5][5];
    GaussianFilterCreation(gaussianKernelData);
    cv::Mat gaussianKernel(5,5, CV_32F, gaussianKernelData);

    convolution(inputImage, outputImage, gaussianKernel);

    //cv::imshow("blurred",outputImage);
    return outputImage;
}

/*
*	Function: secondDerivativeX
	*	Calculates second order derivative in X direction.
	*	The kernel used is a convolution of 1st order sobel in X direction with itself.
	*	InputImage data type - CV_8U, outputImage data type - CV_32F
*/
void secondDerivativeX(const cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_X_direction[] = {1, 0, -2, 0, 1,
    									  4, 0, -8, 0, 4,
    									  6, 0, -12,0, 6,
    									  4, 0, -8, 0, 4,
    									  1, 0, -2, 0, 1 };

    cv::Mat kernel_For_X_direction(5,5,CV_32F,kernelData_For_X_direction);

    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_X_direction);
    
    cv::imshow("Edges along X", gaussianBlur(edgeImage));

}

/*
*	Function: secondDerivativeY
	*	Calculates second order derivative in Y direction.
	*	The kernel used is a convolution of 1st order sobel in Y direction with itself.
	*	InputImage data type - CV_8U, outputImage data type - CV_32F
*/
void secondDerivativeY(const cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_Y_direction[] = {1, 4, 6, 4, 1,
    									  0, 0, 0, 0, 0,
    									  -2, -8, -12, -8 ,-2,
    									  0, 0, 0, 0, 0, 
    									  1, 4, 6 ,4, 1 };

    cv::Mat kernel_For_Y_direction(5,5,CV_32F,kernelData_For_Y_direction);
    
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_Y_direction);
   
    cv::imshow("Edges along Y", gaussianBlur(edgeImage));

}

/*
*	Function: secondDerivativeXY
	*	Calculates second order derivative in X and Y direction.
	*	The kernel is obtained by the convolution of 1st order sobel kernel in X and Y direction
	*	InputImage data type - CV_8U, outputImage data type - CV_32F
*/
void secondDerivativeXY(const cv::Mat& inputImage, cv::Mat& edgeImage){

	float kernelData[] = { 1, 2, 0, -2, -1, 
                    	   2, 4, 0, -4, -2,
                           0, 0, 0, 0, 0,
                       	  -2, -4, 0, 4, 2,
                       	  -1, -2, 0, 2, 1	};

    cv::Mat kernel_For_XY_direction(5,5,CV_32F,kernelData);

    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_XY_direction);
    

    cv::imshow("Edges along XY", gaussianBlur(edgeImage));

}

/*
*	Function: HessianResponse
	*	InputImage, outputImage data type - CV_32F
	*	Calculates hessian determinant. Initializes them in outputImage
*/

void HessianResponse(const cv::Mat& Ixx, const cv::Mat& Iyy, const cv::Mat& Ixy, cv::Mat& outputImage){

	float det_Matrix = 0;

	cv::Mat M(2, 2, CV_32F);

	for(int row=0; row<outputImage.rows; ++row){
		for(int col=0; col<outputImage.cols; ++col){
			M.at<float>(0,0) = (float)Ixx.at<float>(row, col);
			M.at<float>(0,1) = (float)Ixy.at<float>(row, col);
			M.at<float>(1,0) = (float)Ixy.at<float>(row, col);
			M.at<float>(1,1) = (float)Iyy.at<float>(row, col);

			cv::Mat eigenValsMat;
			cv::eigen(M, eigenValsMat);

			det_Matrix = (float)(eigenValsMat.at<float>(0,0) * eigenValsMat.at<float>(1,0));

			outputImage.at<float>(row, col) = det_Matrix;
		}
	}
}

/*
*	Function: findMax
	*	InputImage, outputImage data type - CV_8U
	*	In inputImage, for determinant values > hessianThreshold, initialize 255 in outputImage.
*/
cv::Mat findMax(cv::Mat inputImage){

	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8U);
	
	for(int row = 0; row < inputImage.rows; ++row){
		for(int col = 0; col < inputImage.cols; ++col){

			if(inputImage.at<uint8_t>(row, col) > 100){ //100 is hessian threshold for interest point.
				outputImage.at<uint8_t>(row, col)=255;
			}

			else{
				outputImage.at<uint8_t>(row, col)=0;
			}
		}
	}

	return outputImage;
}

void highlightPoints(cv::Mat& inputImage, cv::Mat& points){ // Points image is binary

	for(int row = 0; row < inputImage.rows; ++row){
		for(int col = 0; col < inputImage.cols; ++col){
			if(points.at<uint8_t>(row, col)>0){
				cv::circle(inputImage, cv::Point(col,row), 3, cv::Scalar(0, 255, 0), 1);
			}
		}
	}
}

void erode(cv::Mat& inputImage, cv::Mat& outputImage){
	
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

void normalize(const cv::Mat& hessianMatInput, cv::Mat& hessianMatNormalized){
	float lowestPixel=0, highestPixel=0;

	lowestPixel = (float)hessianMatInput.at<float>(0,0); //initialize a random point in matrix
	for (int row = 0; row < hessianMatInput.rows; row++){
		for (int col = 0; col < hessianMatInput.cols; col++){
			
			if((float)hessianMatInput.at<float>(row, col)<=lowestPixel){
				lowestPixel = (float)hessianMatInput.at<float>(row, col);
			}

			if((float)hessianMatInput.at<float>(row, col)>=highestPixel){
				highestPixel = (float)hessianMatInput.at<float>(row, col);
			}
		}
	}

	int maxIntensity=255;
	int minIntensity=0;

	for(int row=0;row<hessianMatInput.rows;row++) {
        for(int col = 0; col < hessianMatInput.cols; col++) {
            hessianMatNormalized.at<uint8_t>(row, col) =  (int)(hessianMatInput.at<float>(row, col) - lowestPixel) * ((maxIntensity - minIntensity)/(highestPixel - lowestPixel));
        }
    }
}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat inputImage = cv::imread("hessianInput.png");
	cv::imshow("Input Image", inputImage);

	// Step 1: Convert the BGR image into a grayscale Image.
	cv::Mat grayImage(inputImage.rows, inputImage.cols, CV_8UC1);
	convertToGray(inputImage, grayImage);
	
	//Step 2: Compute Second derivatives along X, Y and XY directions
	cv::Mat Ixx(inputImage.rows, inputImage.cols, CV_32F);
	secondDerivativeX(grayImage, Ixx);

	cv::Mat Iyy(inputImage.rows, inputImage.cols, CV_32F);
	secondDerivativeY(grayImage, Iyy);

	cv::Mat Ixy(inputImage.rows, inputImage.cols, CV_32F);
	secondDerivativeXY(grayImage, Ixy);
	
	// Step 3: Apply formula for Harris
	cv::Mat hessianOutput(inputImage.rows, inputImage.cols, CV_32F);	
	HessianResponse(gaussianBlur(Ixx),gaussianBlur(Iyy),gaussianBlur(Ixy), hessianOutput);
	//std::cout << hessianOutput << std::endl;
	cv::Mat normalizedHessian = cv::Mat::zeros(hessianOutput.rows, hessianOutput.cols, CV_8U);
	normalize(hessianOutput, normalizedHessian);
	
	// Step 4: Find the points of interest
	cv::Mat PointsImage = findMax(normalizedHessian);

	//Draw circles on the input image based on the points detected
	highlightPoints(inputImage, PointsImage);
	cv::imshow("Image with corners highlighted", inputImage);

	saveImageToFile(inputImage, "hessianOutputImage.jpg");
	cv::waitKey(0);
	cv::destroyAllWindows();


	return 0;

}
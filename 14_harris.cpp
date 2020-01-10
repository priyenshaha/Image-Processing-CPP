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

//This function creates a 5 x 5 gaussian kernel
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

int convolution_for_gaussian(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
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

    convolution_for_gaussian(inputImage, outputImage, gaussianKernel);

    //cv::imshow("blurred",outputImage);
    return outputImage;
}

/*
*	Function: edgesX
	*	InputImage data type- CV_8U, outputImage data type - CV_32F
	*	3 x 3 kernel, data type - CV_32F
	*	Detects edges along X direction
*/
void edgesX(const cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_X_direction[] = {-1, 0, 1, 
                                          -2, 0, 2,
                                          -1, 0, 1};

    cv::Mat kernel_For_X_direction(3,3,CV_32F,kernelData_For_X_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_X_direction);

    //cv::imshow("Edges along X", edgeImage);

}

/*
*	Function: edgesY
	*	InputImage data type- CV_8U, outputImage data type - CV_32F
	*	3 x 3 kernel, data type - CV_32F
	*	Detects edges along Y direction
*/
void edgesY(const cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_Y_direction[] = {-1, -2, -1, 
                                           0,  0,  0,
                                           1,  2,  1 };

    cv::Mat kernel_For_Y_direction(3,3,CV_32F,kernelData_For_Y_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_Y_direction);

    //cv::imshow("Edges along Y", edgeImage);

}

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

/*
*	Function: HarrisResponse
	*	InputImage, outputImage data type - CV_32F
	*	Calculates R values. Initializes them in outputImage
*/
void HarrisResponse(const cv::Mat& Ix, const cv::Mat& Iy, cv::Mat& outputImage){

	float k = 0.05;	//empirically calculated value. k=[0,04, 0,06]

	cv::Mat Ixx = cv::Mat::zeros(Ix.rows, Ix.cols, CV_32F);
	cv::Mat Iyy = cv::Mat::zeros(Ix.rows, Ix.cols, CV_32F);
	cv::Mat Ixy = cv::Mat::zeros(Ix.rows, Ix.cols, CV_32F);

	float det_Matrix = 0, trace_Matrix = 0;

	for(int row = 0; row < Ix.rows; row++){
		for(int col = 0; col < Ix.cols; col++){

			Ixx.at<float>(row, col) = (float)(pow(Ix.at<float>(row, col), 2));
			Iyy.at<float>(row, col) = (float)(pow(Iy.at<float>(row, col), 2));
			Ixy.at<float>(row, col) = (float)((Ix.at<float>(row, col))*(Iy.at<float>(row, col)));

		}
	}

	Ixx = gaussianBlur(Ixx);
	Iyy = gaussianBlur(Iyy);
	Ixy = gaussianBlur(Ixy);

/*
	M = [Ixx, Ixy]
		[Ixy, Iyy]

	det = Ixx * Iyy - (Ixy)^2

	trace = Ixx + Iyy

	R = det - k*(trace^2)
*/

	for(int row=0; row<outputImage.rows; ++row){
		for(int col=0; col<outputImage.cols; ++col){

			det_Matrix = (float)((Ixx.at<float>(row, col) * Iyy.at<float>(row, col)) - (pow(Ixy.at<float>(row, col), 2)));
			
			trace_Matrix = (float)(Ixx.at<float>(row, col) + Iyy.at<float>(row, col));

			outputImage.at<float>(row, col) = (float)(det_Matrix - k*(pow(trace_Matrix, 2)));
		}
	}
}

/*
*	Function: findMax
	*	InputImage data type- CV_32F, outputImage data type - CV_8U
	*	In inputImage, for R > 0 (corners), initialize 255 in outputImage.
*/
cv::Mat findMax(cv::Mat inputImage){

	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8U);


	for(int row = 0; row < inputImage.rows; ++row){
		for(int col = 0; col < inputImage.cols; ++col){

			//Consider only positive values
			if(inputImage.at<float>(row, col) > 0){
				outputImage.at<uchar>(row, col)=255;
			}

			else{
				outputImage.at<uchar>(row, col)=0;
			}
		}
	}

	return outputImage;
}

/*
*	Function: highlightPoints
	*	Draw coloured circles corresponding to the points detected from eroding the harris response output.
	*	3 x 3 kernel, data type - CV_32F
*/
void highlightPoints(cv::Mat& colorImage, cv::Mat& points){

	for(int row = 0; row < colorImage.rows; ++row){
		for(int col = 0; col < colorImage.cols; ++col){
			if(points.at<uint8_t>(row, col)>127){
				cv::circle(colorImage, cv::Point(col,row), 3, cv::Scalar(0, 0, 255), 1);
			}
		}
	}
}

/*
*	Function: erode
	*	InputImage, outputImage data type- CV_8U
	*	Erodes the image with kernel: 	[0, 1, 0]
										[1, 1, 1]
										[0, 1, 0]

	*	Used this function to precisely locate the corners.
*/
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

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

	cv::Mat inputImage = cv::imread("chess.png");

	//cv::imshow("Input Image", inputImage);

	// Step 1: Convert the BGR image into a grayscale Image.
	cv::Mat grayImage(inputImage.rows, inputImage.cols, CV_8UC1);
	convertToGray(inputImage, grayImage);

	//Step 2A: Detect Edges along X direction. That is the first order derivative along X direction
	cv::Mat Ix(inputImage.rows, inputImage.cols, CV_32F);
	edgesX(grayImage, Ix);
	
	//Step 2B: Detect Edges along Y direction. That is the first order derivative along Y direction
	cv::Mat Iy(inputImage.rows, inputImage.cols, CV_32F);
	edgesY(grayImage, Iy);
	
	// Step 3: Apply formula for Harris
	cv::Mat harrisOutput(inputImage.rows, inputImage.cols, CV_32F);	
	HarrisResponse(Ix, Iy, harrisOutput);

	// Step 4: Find the corner points. Here, points such that R > 0 for corners.
	cv::Mat cornerPointsImage = findMax(harrisOutput);//, inputImage);
	//cv::imshow("Corner Points harrisOutput", cornerPointsImage);

	//Step 5: Erode the image with corners to get finer corner points
	cv::Mat erodedCorners = cornerPointsImage.clone();
	erode(cornerPointsImage, erodedCorners);
	cv::imshow("Detected Corners", erodedCorners);

	//Draw circles on the input image based on the points detected
	highlightPoints(inputImage, erodedCorners);
	cv::imshow("Image with highlighted corners", inputImage);

	saveImageToFile(inputImage, "harrisCornerDetectionOutputImage.jpg")
	cv::waitKey(0);
	cv::destroyAllWindows();


	return 0;

}
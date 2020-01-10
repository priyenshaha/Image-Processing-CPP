/*
*   Author - Priyen Shah
*   Emp ID - 141541

Problem statement:

 Write a program for canny edge detection. 
 Its a process of 4 modules viz. 
 i) Gaussian smoothing 
 ii) Sobel Edge detection 
 iii) non-maximal suppression 
 iv) Hysteresis Thresholding

*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <cmath>

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

int convolution_for_8_bit_mat(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
    //Create a inputImage object of all zeros. The data type should be same as input image. Here unsigned int of 8 bits
    cv::Mat paddedImage = cv::Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_8U);

    //Padding of image
    //Add the input image in to the paddedImage inputImage obj. Leave the size of padding aside. So initialize the row and col counter accordingly
    for(int row=(int)kernel.rows/2;row<paddedImage.rows-(int)kernel.rows/2;row++) {
        for(int col=(int)kernel.cols/2;col<paddedImage.cols-(int)kernel.cols/2;col++) {
            // Ensure that the input image data is taken from zeroth index.
            paddedImage.at<uint8_t>(row, col) = inputImage.at<uint8_t>(row-(int)kernel.rows/2, col-(int)kernel.cols/2);
        }
    }

    //Convolve the image with the input filter
    for(int row=(int)kernel.rows/2; row<((paddedImage.rows)-(int)kernel.rows/2); row++) {
        for(int col=(int)kernel.cols/2; col<((paddedImage.cols)-(int)kernel.cols/2); col++) {
            int t_sum = 0;
            for(int k_row=-(int)kernel.rows/2; k_row<(kernel.rows-(int)kernel.rows/2); k_row++) {
                for(int k_col=-(int)kernel.cols/2; k_col<(kernel.cols-(int)kernel.cols/2); k_col++) {
                    t_sum += (int)((int)paddedImage.at<uint8_t>(row+k_row, col+k_col) * (float)kernel.at<float>(k_row+(int)kernel.rows/2, k_col+(int)kernel.cols/2));
                }
            }
            outputImage.at<uint8_t>(row-(int)kernel.rows/2, col-(int)kernel.cols/2) = t_sum;
        }
    }
}

int convolution_for_32_bit_mat(cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& kernel) {
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

void gaussianBlur(cv::Mat& inputImage, cv::Mat& outputImage){

    float gaussianKernelData[5][5];
    GaussianFilterCreation(gaussianKernelData);
    cv::Mat gaussianKernel(5,5, CV_32F, gaussianKernelData);

    convolution_for_8_bit_mat(inputImage, outputImage, gaussianKernel);

    cv::imshow("blurred",outputImage);
}

void edgesX(cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_X_direction[] = {-1, 0, 1, 
                                          -2, 0, 2,
                                          -1, 0, 1};

    cv::Mat kernel_For_X_direction(3,3,CV_32F,kernelData_For_X_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_X_direction);

    cv::imshow("Edges along X", edgeImage);

}

void edgesY(cv::Mat& inputImage, cv::Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_Y_direction[] = {-1, -2, -1, 
                                           0,  0,  0,
                                           1,  2,  1 };

    cv::Mat kernel_For_Y_direction(3,3,CV_32F,kernelData_For_Y_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_Y_direction);

    cv::imshow("Edges along Y", edgeImage);

}

cv::Mat computeGradientMagnitude(cv::Mat& edges_along_X, cv::Mat& edges_along_Y){
    //Initialize this matrix object as CV_32F as the calculated values will be floating in nature.
    cv::Mat gradientMatrix = cv::Mat::zeros(edges_along_X.rows, edges_along_X.cols, CV_32F);

    //Calculation of Gradient Magnitude    
    for(int row=0; row<gradientMatrix.rows ;row++) {
        for(int col=0;col<gradientMatrix.cols;col++) {
            gradientMatrix.at<float>(row, col) = (float)sqrt(pow(edges_along_X.at<float>(row, col), 2) 
                                                          + pow(edges_along_Y.at<float>(row, col), 2));
        }
    }

    //initialize some value as max value
    float maxValue = gradientMatrix.at<float>(0, 0); 
    for(int row=0;row<gradientMatrix.rows;row++) {
        for(int col=0;col<gradientMatrix.cols;col++) {

            //Update max value if larger value found
            if(gradientMatrix.at<float>(row, col)>maxValue) {
                maxValue = gradientMatrix.at<float>(row, col);
            }
        }
    }

    //Normalize the entire image w.r.t max value to  retain the edges only.
    for(int row=0;row<gradientMatrix.rows;row++) {
        for(int col=0; col<gradientMatrix.cols; col++) {
            gradientMatrix.at<float>(row, col) *= 255/maxValue;      
        }
    }


    for(int row=0;row<gradientMatrix.rows;row++) {
        for(int col=0;col<gradientMatrix.cols;col++) {
            gradientMatrix.at<float>(row, col) = (int)gradientMatrix.at<float>(row, col);      
        }
    }

    cv::Mat output(gradientMatrix.rows,gradientMatrix.cols, CV_8U, cv::Scalar::all(0));

    //Covert the CV_32F image to CV_8U
    for(int row=0;row<gradientMatrix.rows;row++) {
        for(int col=0;col<gradientMatrix.cols;col++) {
            output.at<uchar>(row, col) = gradientMatrix.at<float>(row, col);      
        }
    }
    return output;

}

void saveImageToFile(cv::Mat& image, std::string fileName){

    cv::imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

int main(){

    cv::Mat inputImage = cv::imread("fish.png",0);
    cv::imshow("Input image", inputImage);

    cv::Mat blurredImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);

    gaussianBlur(inputImage, blurredImage);

    cv::Mat edges_along_X = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    cv::Mat edges_along_Y = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);

    edgesX(blurredImage, edges_along_X);
    edgesY(blurredImage, edges_along_Y);

    cv::Mat sobelOutput = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
    sobelOutput = computeGradientMagnitude(edges_along_X, edges_along_Y);
    cv::imshow("Sobel Output", sobelOutput);

    saveImageToFile(sobelOutput, "sobelEdgeOutput.jpg");

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

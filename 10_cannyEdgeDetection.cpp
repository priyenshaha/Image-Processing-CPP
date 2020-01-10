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

double pi(){ 
    return std::atan(1)*4; 
}

cv::Mat gradientDirection(cv::Mat& edges_along_X, cv::Mat& edges_along_Y) {

    cv::Mat gradAngle = cv::Mat::zeros(edges_along_X.rows, edges_along_X.cols, CV_32F);

    //Calculation of gradient direction in radians
    for(int row=0;row < gradAngle.rows;row++) {
        for(int col=0;col<gradAngle.cols;col++) {
            gradAngle.at<float>(row, col) = atan2(edges_along_Y.at<float>(row, col), edges_along_X.at<float>(row, col));
        }
    }

    //Conversion from radians to degrees
    for(int row=0;row<gradAngle.rows;row++) {
        for(int col=0;col<gradAngle.cols;col++) {
            gradAngle.at<float>(row, col) = (180 * gradAngle.at<float>(row, col))/pi();
        }
    }

    // Setting to 0(degree) to 180(degree)
    for(int row=0;row<gradAngle.rows;row++) {
        for(int col=0;col<gradAngle.cols;col++) {
            gradAngle.at<float>(row, col) += 180 ;
        }
    }

    return gradAngle;
}

cv::Mat non_max_suppression(cv::Mat& magnitude, cv::Mat& direction) {

    cv::Mat suppressed = cv::Mat::zeros(magnitude.rows,magnitude.cols, CV_8U);
    int before_pixel, after_pixel;
    int PI=180;

    for(int row=1;row<(magnitude.rows-1);row++) {
        for(int col=1;col<(magnitude.cols-1);col++) {

           //East-West direction
           if((0 <= direction.at<float>(row,col) < (PI/8)) || ((15*PI/8) <= direction.at<float>(row,col) <= (2*PI)) || (157.5 <= direction.at<float>(row,col) < 202.5)) {
                before_pixel = magnitude.at<uint8_t>(row, col-1); 
                after_pixel = magnitude.at<uint8_t>(row, col+1); 
           }

           //NorthEast-SouthWest direction
           else if(((PI/8) <= direction.at<float>(row,col) < (3*PI/8)) || ((9*PI/8) <= direction.at<float>(row,col) < (11*PI/8))) {
                before_pixel = magnitude.at<uint8_t>(row+1, col-1); 
                after_pixel = magnitude.at<uint8_t>(row-1, col+1);    
           }

           //North-South direction
           else if(((3*PI/8) <= direction.at<float>(row,col) < (5*PI/8)) || ((11*PI/8) <= direction.at<float>(row,col) < (13*PI/8))) {
                before_pixel = magnitude.at<uint8_t>(row-1, col); 
                after_pixel = magnitude.at<uint8_t>(row+1, col);    
           }

           //(5*PI/8, 7*PI/8) or (13*PI/8, 15*PI/8) -> NorthWest-SouthEast direction
           else {
                before_pixel = magnitude.at<uint8_t>(row-1, col-1); 
                after_pixel = magnitude.at<uint8_t>(row+1, col+1);    
           }

           if((magnitude.at<uint8_t>(row, col)>=before_pixel) && (magnitude.at<uint8_t>(row, col)>=after_pixel)) {
               suppressed.at<uint8_t>(row, col) = magnitude.at<uint8_t>(row, col); 
           }
        
           else {
               suppressed.at<uint8_t>(row, col) = 0; 
           }
        }
    }

    return suppressed;
}

cv:: Mat threshold(cv::Mat& non_max, int low, int high, int weak) {

    cv::Mat thresh = cv::Mat::zeros(non_max.rows,non_max.cols, CV_8U);

    int strong = 255;

    for(int row=0;row<non_max.rows;row++) {
        for(int col=0;col<non_max.cols;col++) {
            if(non_max.at<uint8_t>(row, col)>high) {
                thresh.at<uint8_t>(row, col) = strong;
            }
            else if((non_max.at<uint8_t>(row, col)<=high)&&(non_max.at<uint8_t>(row, col)>=low)) {
                thresh.at<uint8_t>(row, col) = weak;
            }
       }
    }
    return thresh;
}

cv::Mat hysteresis(cv::Mat& doubleThresholdedImage, int weak) {

    cv::Mat top_bottom = doubleThresholdedImage.clone();
    cv::Mat bottom_top = doubleThresholdedImage.clone();
    cv::Mat right_left = doubleThresholdedImage.clone();
    cv::Mat left_right = doubleThresholdedImage.clone();


    //Scanning Top to bottom
    for(int row=1;row<doubleThresholdedImage.rows;row++) {
        for(int col=1;col<doubleThresholdedImage.cols;col++) {
            if(top_bottom.at<uint8_t>(row, col)==weak) {
                if((top_bottom.at<uint8_t>(row, col+1) == 255) || (top_bottom.at<uint8_t>(row, col-1) == 255) || (top_bottom.at<uint8_t>(row-1, col) == 255) || (top_bottom.at<uint8_t>(row+1, col) == 255) || (top_bottom.at<uint8_t>(row-1, col-1) == 255) || (top_bottom.at<uint8_t>(row+1, col-1) == 255) || (top_bottom.at<uint8_t>(row-1, col+1) == 255) || (top_bottom.at<uint8_t>(row+1, col+1) == 255)) {
                    top_bottom.at<uint8_t>(row, col) = 255;
                }
                else {
                    top_bottom.at<uint8_t>(row, col) = 0;
                }
             }
         }
    }

    //Scanning Bottom to Top
    for(int row=(doubleThresholdedImage.rows-1);row>0;row--) {
        for(int col=(doubleThresholdedImage.cols-1);col>0;col--) {
            if(bottom_top.at<uint8_t>(row, col)==weak) {
                if((bottom_top.at<uint8_t>(row, col+1) == 255) || (bottom_top.at<uint8_t>(row, col-1) == 255) || (bottom_top.at<uint8_t>(row-1, col) == 255) || (bottom_top.at<uint8_t>(row+1, col) == 255) || (bottom_top.at<uint8_t>(row-1, col-1) == 255) || (bottom_top.at<uint8_t>(row+1, col-1) == 255) || (bottom_top.at<uint8_t>(row-1, col+1) == 255) || (bottom_top.at<uint8_t>(row+1, col+1) == 255)) {
                    bottom_top.at<uint8_t>(row, col) = 255;
                }
                else {
                    bottom_top.at<uint8_t>(row, col) = 0;
                }
             }
         }
    }
  
    //Scanning Right to Left
    for(int row=1;row<doubleThresholdedImage.rows;row++) {
        for(int col=(doubleThresholdedImage.cols-1);col>0;col--) {
            if(right_left.at<uint8_t>(row, col)==weak) {
                if((right_left.at<uint8_t>(row, col+1) == 255) || (right_left.at<uint8_t>(row, col-1) == 255) || (right_left.at<uint8_t>(row-1, col) == 255) || (right_left.at<uint8_t>(row+1, col) == 255) || (right_left.at<uint8_t>(row-1, col-1) == 255) || (right_left.at<uint8_t>(row+1, col-1) == 255) || (right_left.at<uint8_t>(row-1, col+1) == 255) || (right_left.at<uint8_t>(row+1, col+1) == 255)) {
                    right_left.at<uint8_t>(row, col) = 255;
                }
                else {
                    right_left.at<uint8_t>(row, col) = 0;
                }
             }
         }
    }

    //Scanning Left to Right
    for(int row=(doubleThresholdedImage.rows-1);row>0;row--) {
        for(int col=1;col<doubleThresholdedImage.cols;col++) {
            if(left_right.at<uint8_t>(row, col)==weak) {
                if((left_right.at<uint8_t>(row, col+1) == 255) || (left_right.at<uint8_t>(row, col-1) == 255) || (left_right.at<uint8_t>(row-1, col) == 255) || (left_right.at<uint8_t>(row+1, col) == 255) || (left_right.at<uint8_t>(row-1, col-1) == 255) || (left_right.at<uint8_t>(row+1, col-1) == 255) || (left_right.at<uint8_t>(row-1, col+1) == 255) || (left_right.at<uint8_t>(row+1, col+1) == 255)) {
                    left_right.at<uint8_t>(row, col) = 255;
                }
                else {
                    left_right.at<uint8_t>(row, col) = 0;
                }
             }
         }
    }

    cv::Mat final_image = cv::Mat::zeros(doubleThresholdedImage.rows,doubleThresholdedImage.cols, CV_32F);

    for(int row=0;row<doubleThresholdedImage.rows;row++) {
        for(int col=0;col<doubleThresholdedImage.cols;col++) {
            final_image.at<float>(row, col) = (float)(top_bottom.at<uint8_t>(row, col) + bottom_top.at<uint8_t>(row, col)+right_left.at<uint8_t>(row, col)+left_right.at<uint8_t>(row, col));
        }
    }

    for(int row=0;row<doubleThresholdedImage.rows;row++) {
        for(int col=0;col<doubleThresholdedImage.cols;col++) {
            if(final_image.at<float>(row, col)>255) {
                final_image.at<float>(row, col) = 255;
            }
        }
    }

    cv::Mat output = cv::Mat::zeros(doubleThresholdedImage.rows,doubleThresholdedImage.cols, CV_8U);

    //Converison to 8 bit
    for(int row=0;row<doubleThresholdedImage.rows;row++) {
        for(int col=0;col<doubleThresholdedImage.cols;col++) {
            output.at<uint8_t>(row, col) = final_image.at<float>(row, col);      
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

    cv::Mat gradientMagnitude = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
    gradientMagnitude = computeGradientMagnitude(edges_along_X, edges_along_Y);
    cv::imshow("Gradient Magnitude (Sobel Edges)", gradientMagnitude);

    cv::Mat gradientAngle = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    gradientAngle = gradientDirection(edges_along_X, edges_along_Y);

    cv::Mat non_max = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    non_max = non_max_suppression(gradientMagnitude, gradientAngle);

    cv::Mat doubleThresholdedImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    int weakPixelVal=50;

    doubleThresholdedImage = threshold(non_max, 10, 25, weakPixelVal);
    cv::imshow( "Double Threshold", doubleThresholdedImage);

    cv::Mat cannyImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    cannyImage = hysteresis(doubleThresholdedImage, weakPixelVal);
    cv::imshow("Canny Edge output", cannyImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    saveImageToFile(cannyImage, "cannyEdgeOutput.jpg");

    return 0;
}

#include <iostream>
#include <cmath>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

#define BIN_WIDTH 1                 
#define NUM_BINS 180 / BIN_WIDTH   

/* === PARAMS FOR CANNY EDGE DETECTION === */

#define KERNEL_SIZE 3
#define THRESHOLD 50
#define RATIO 3

//This function implements canny edge detection and initializes an image with edges. Input image is a grayscale
void detectEdge(Mat& in, Mat& out);

//This funtion is used to convert the polar values (r, theta) to cartesian coordinates and initializes 2 points to draw a line
void polarToCartesian(double rho, int theta, Point& p1, Point& p2, int maxDistance);

//These functions are used to implement canny edge detection algorithm
void GaussianFilterCreation(float GKernel[][5]);
int convolution_for_8_bit_mat(Mat& inputImage, Mat& outputImage, Mat& kernel);
int convolution_for_32_bit_mat(Mat& inputImage, Mat& outputImage, Mat& kernel);
void gaussianBlurring(Mat& inputImage, Mat& outputImage);
void edgesX(Mat& inputImage, Mat& edgeImage);
void edgesY(Mat& inputImage, Mat& edgeImage);
Mat computeGradientMagnitude(Mat& edges_along_X, Mat& edges_along_Y);
double pi();
Mat gradientDirection(Mat& edges_along_X, Mat& edges_along_Y);
Mat non_max_suppression(Mat& magnitude, Mat& direction);
Mat threshold(Mat& non_max, int low, int high, int weak);
Mat hysteresis(Mat& doubleThresholdedImage, int weak);

//===================================================================

//This function save the output to an image file.
void saveImageToFile(Mat& image, std::string fileName);

//This function is used to draw finite lines. The line connecting p1 and p2 is iterated to find the presense of object in edgeImage.
//The parameters requierd are minimum line length that can be drawn and maximum gap between two objects along a same line
void drawFiniteLine(Mat edgeImage, Mat& outputImage, Point& p1, Point& p2, int minLength, int maxGap);

//This function calculates the euclidiean distance between the 2 points.
int getDistance(Point p1, Point p2);

//Converts BGR image into grayscale image
void convertToGray(const cv::Mat& inputImage, cv::Mat& outputImage);
/*=========================================================*/

int main(int argc, char** argv) {

    int i, j;
    int theta;      // parameter for cartesian to polar
    double rho;     // distance parameter

    Mat input;

    //Accept inputs from the command line
    if(argc < 5) {
        printf("USAGE: ./houghExec [fileName] [line threshold] [min line length] [max gap in lines] \n");
        return EXIT_FAILURE;
    }

    int lineThreshold = atoi(argv[2]);
    int minLength = atoi(argv[3]);
    int maxGap = atoi(argv[4]);

    input = imread(argv[1]);  //Color image input to plot colored line in output.

    //Declare the Mat objects 
    Mat grayImage = Mat::zeros(input.rows, input.cols, CV_8UC1);
    Mat edges = Mat::zeros(grayImage.rows, grayImage.cols, CV_8UC1);
    Mat finiteOutput = Mat::zeros(grayImage.rows, grayImage.cols, CV_8UC1);
    Mat infiniteOutput = Mat::zeros(grayImage.rows, grayImage.cols, CV_8UC1);

    //Convert the BGR image to grayscale.
    convertToGray(input, grayImage);

    //Compute the length of diagonal
    int maxDistance = hypot(grayImage.rows, grayImage.cols);

    // matrix of votes
    Mat votes = Mat::zeros(2 * maxDistance, NUM_BINS, CV_8U);

    detectEdge(grayImage, edges);

    // vote
    for(i = 0; i < edges.rows; ++i) {
        for(j = 0; j < edges.cols; ++j) {

            if(edges.at<uchar>(i, j) == 255) {

                // limits for theta: [-90, 90]
                for(theta = 0; theta <= 180; theta += BIN_WIDTH) {

                    rho = round(j * cos((theta - 90)*pi()/180) + i * sin((theta - 90)*pi()/180)) + maxDistance;
                    //cout << rho << " ";
                    votes.at<uchar>(rho,theta)+=1;
                }
            }
        }
    }

    infiniteOutput = input.clone();
    finiteOutput = input.clone();
    // find peaks
    for(i = 0; i < votes.rows; ++i) {
        for(j = 0; j < votes.cols; ++j) {

            if(votes.at<uchar>(i,j) >= lineThreshold) {

                rho = i - maxDistance;
                theta = j - 90;
                //cout << "found line with rho = " << rho << " and theta = " << theta << "\n";
                
                // Convert the polar form of line into cartesian
                Point p1, p2;        // 2 points to describe the line

                polarToCartesian(rho, theta, p1, p2, maxDistance);
                cout << "P1: " << p1 << ", P2: " << p2 << endl;

                drawFiniteLine(edges, finiteOutput, p1, p2, minLength, maxGap);

                line(infiniteOutput, p1, p2, Scalar(0, 0, 255), 1, LINE_AA);

            }
        }
    }

    imshow("grayImage image", grayImage);
    imshow("Edge detector output", edges);
    imshow("infinite line output image", infiniteOutput);
    imshow("finite line output image", finiteOutput);

    saveImageToFile(finiteOutput, "../../outputImages/houghOutput.jpg");

    waitKey(0);


    return 0;
}

void detectEdge(Mat& in, Mat& out) {

  blur(in, out, Size(3, 3));  

  Canny(out, out, 80, 240, KERNEL_SIZE);
}

void drawFiniteLine(Mat edgeImage, Mat& outputImage, Point& p1, Point& p2, int minLength, int maxGap){

  int currentLength=0, currentGap=0, startFlag=0, stopFlag=0;
  Point QStart, QStop, Qprev;

  LineIterator it(edgeImage, p1, p2);

  for(int i=0; i < it.count; ++i, ++it){

    if(i==0)
      continue;

    int pixelValue = (int)edgeImage.at<uchar>(it.pos());
    //cout << "[" << it.pos().x-1 << ", " << it.pos().y << "], " << "Value: " << pixelValue;

    if(pixelValue!=0){
      //cout << "\nFound a pixel: "  << it.pos() << ", Value: " << pixelValue << "currentLength: " << currentLength << endl;

      if(startFlag==0){

        QStart = it.pos();
        Qprev = QStart;
        startFlag = 1;
      }
      
      cout << "Qstart: "<< QStart <<", Qprev: " << Qprev << "Current pos: " << it.pos() << endl;

      if(startFlag==1)
      {
        if((getDistance(Qprev, it.pos()))>maxGap){
          
          QStop = Qprev;
          startFlag=0;

          if(i==it.count)
            QStop = Qprev;

          if(getDistance(QStart, QStop)>=minLength){

            line(outputImage, QStart, QStop, Scalar(0, 0, 255), 1, LINE_AA);

          }

        }

      }

      Qprev = it.pos();
     
    }
  }
}

int getDistance(Point p1, Point p2){

  int x1 = p1.x, x2 = p2.x, y1 = p1.y, y2 = p2.y;

  return (sqrt(pow((x2-x1), 2) + pow((y2-y1), 2)));
}


void polarToCartesian(double rho, int theta, Point& p1, Point& p2, int maxDistance) {

  int x0 = cvRound(rho * cos(theta*pi()/180));
  int y0 = cvRound(rho * sin(theta*pi()/180));

  p1.x = cvRound(x0 + maxDistance * (-sin(theta*pi()/180)));
  p1.y = cvRound(y0 + maxDistance * (cos(theta*pi()/180)));

  p2.x = cvRound(x0 - maxDistance * (-sin(theta*pi()/180)));
  p2.y = cvRound(y0 - maxDistance * (cos(theta*pi()/180)));
}

/*
* Function: convertToGray
  * InputImage data type- CV_8UC3, outputImage data type - CV_8UC1
  * Converts BGR image into grayscale image
  
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

/*void detectEdge(Mat& inputImage, Mat& outputImage){

  Mat blurredImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);

  gaussianBlurring(inputImage, blurredImage);

  Mat edges_along_X = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
  Mat edges_along_Y = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);

  edgesX(blurredImage, edges_along_X);
  edgesY(blurredImage, edges_along_Y);

  Mat gradientMagnitude = Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
  gradientMagnitude = computeGradientMagnitude(edges_along_X, edges_along_Y);
  //imshow("Gradient Magnitude (Sobel Edges)", gradientMagnitude);

  Mat gradientAngle = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
  gradientAngle = gradientDirection(edges_along_X, edges_along_Y);

  Mat non_max = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
  non_max = non_max_suppression(gradientMagnitude, gradientAngle);

  Mat doubleThresholdedImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
  int weakPixelVal=10;

  doubleThresholdedImage = threshold(non_max, 20, 80, weakPixelVal);
  //imshow( "Double Threshold", doubleThresholdedImage);

  outputImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
  outputImage = hysteresis(doubleThresholdedImage, weakPixelVal);
  
  imshow("Canny Edge output", outputImage);

}*/

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

int convolution_for_8_bit_mat(Mat& inputImage, Mat& outputImage, Mat& kernel) {
    //Create a inputImage object of all zeros. The data type should be same as input image. Here unsigned int of 8 bits
    Mat paddedImage = Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_8U);

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

int convolution_for_32_bit_mat(Mat& inputImage, Mat& outputImage, Mat& kernel) {
    //Create a inputImage object of all zeros. The data type should be same as input image. Here unsigned int of 32 bits
    Mat paddedImage = Mat::zeros(inputImage.rows+kernel.rows-1,inputImage.cols+kernel.cols-1, CV_32F);

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

void gaussianBlurring(Mat& inputImage, Mat& outputImage){

    float gaussianKernelData[5][5];
    GaussianFilterCreation(gaussianKernelData);
    Mat gaussianKernel(5,5, CV_32F, gaussianKernelData);

    convolution_for_8_bit_mat(inputImage, outputImage, gaussianKernel);

    //imshow("blurred",outputImage);
}

void edgesX(Mat& inputImage, Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_X_direction[] = {-1, 0, 1, 
                                          -2, 0, 2,
                                          -1, 0, 1};

    Mat kernel_For_X_direction(3,3,CV_32F,kernelData_For_X_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_X_direction);

    //imshow("Edges along X", edgeImage);

}

void edgesY(Mat& inputImage, Mat& edgeImage){

    //Applying edge detection kernel for X direction
    float kernelData_For_Y_direction[] = {-1, -2, -1, 
                                           0,  0,  0,
                                           1,  2,  1 };

    Mat kernel_For_Y_direction(3,3,CV_32F,kernelData_For_Y_direction);
    convolution_for_32_bit_mat(inputImage, edgeImage, kernel_For_Y_direction);

    //imshow("Edges along Y", edgeImage);

}

Mat computeGradientMagnitude(Mat& edges_along_X, Mat& edges_along_Y){
    //Initialize this matrix object as CV_32F as the calculated values will be floating in nature.
    Mat gradientMatrix = Mat::zeros(edges_along_X.rows, edges_along_X.cols, CV_32F);

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

    Mat output(gradientMatrix.rows,gradientMatrix.cols, CV_8U, Scalar::all(0));

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

Mat gradientDirection(Mat& edges_along_X, Mat& edges_along_Y) {

    Mat gradAngle = Mat::zeros(edges_along_X.rows, edges_along_X.cols, CV_32F);

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

Mat non_max_suppression(Mat& magnitude, Mat& direction) {

    Mat suppressed = Mat::zeros(magnitude.rows,magnitude.cols, CV_8U);
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

 Mat threshold(Mat& non_max, int low, int high, int weak) {

    Mat thresh = Mat::zeros(non_max.rows,non_max.cols, CV_8U);

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

Mat hysteresis(Mat& doubleThresholdedImage, int weak) {

    Mat top_bottom = doubleThresholdedImage.clone();
    Mat bottom_top = doubleThresholdedImage.clone();
    Mat right_left = doubleThresholdedImage.clone();
    Mat left_right = doubleThresholdedImage.clone();


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

    Mat final_image = Mat::zeros(doubleThresholdedImage.rows,doubleThresholdedImage.cols, CV_32F);

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

    Mat output = Mat::zeros(doubleThresholdedImage.rows,doubleThresholdedImage.cols, CV_8U);

    //Converison to 8 bit
    for(int row=0;row<doubleThresholdedImage.rows;row++) {
        for(int col=0;col<doubleThresholdedImage.cols;col++) {
            output.at<uint8_t>(row, col) = final_image.at<float>(row, col);      
        }
    }
    return output;
}

void saveImageToFile(Mat& image, std::string fileName){

    imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}
#include <iostream>
#include <cmath>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

/* === PARAMS & FUNCTIONS FOR CANNY EDGE DETECTION === */

#define KERNEL_SIZE 3
#define THRESHOLD 50
#define RATIO 3

void extractPointsFromImages(Mat inputImage, vector<int>& x, vector<int>& y);
void gaussianElimination();
void detectEdge(Mat& in, Mat& out);
void printVectors(const vector<int>& x, const vector<int>& y);
void createCoeffMat(int n, Mat& coeffMat, const vector<int>& x, const vector<int>& y);
void createTargetMat(Mat& targetMat, const vector<int>& x, const vector<int>& y);
void convertToGray(const cv::Mat& inputImage, cv::Mat& outputImage);
void saveImageToFile(Mat& image, std::string fileName);

int main(int argc, char** argv) {

	
    if(argc < 2) {
        printf("USAGE: ./curveFit [image path] \n");
        return EXIT_FAILURE;
    }

    Mat inputImage = imread(argv[1]);
    imshow("Input Image", inputImage);

    Mat grayImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
    Mat edgeImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);
    Mat curveImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);

    convertToGray(inputImage, grayImage);

    detectEdge(grayImage, edgeImage);
    imshow("Edge image", edgeImage);

    waitKey(0);

    vector<int> x, y;
    extractPointsFromImages(edgeImage, x, y);

    x.shrink_to_fit();
    y.shrink_to_fit();

    //printVectors(x, y);

    Mat coeffMat = Mat::zeros(3, 3, CV_32F);
    Mat paramMat = Mat::zeros(3, 1, CV_32F);
    Mat targetMat = Mat::zeros(3, 1, CV_32F);

    createCoeffMat(x.size(), coeffMat, x, y);
    createTargetMat(targetMat, x, y);

    paramMat = coeffMat.inv() * targetMat;

    cout << "\nParameters for curve equation [a0, a1, a2]: \n" << paramMat << endl;
    
    vector<Point2f> curvePoints;

    for(int X=0; X < curveImage.cols; ++X){

    	int Y = (int)((float)paramMat.at<float>(0,0) + ((X)*((float)paramMat.at<float>(0,1)) + ((pow(X,2)*(float)paramMat.at<float>(0,2)))));
    	
    	//cout << "[" << X << ", " << Y << "] ";
    	Point2f new_point = Point2f(Y, X);                  //resized to better visualize
        curvePoints.push_back(new_point);                       //add point to vector/list
        
    }

    for (int i = 0; i < curvePoints.size() - 1; i++)        
        line(inputImage, curvePoints[i], curvePoints[i + 1], Scalar(255), 2, CV_AA);

    imshow("Curve image", inputImage);

    saveImageToFile(inputImage, "../../outputImages/curveFittingOutput.jpg");

    waitKey(0);
	return 0;
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
void saveImageToFile(Mat& image, std::string fileName){

    imwrite(fileName, image);
    std::cout << "Output image saved as: " << fileName << std::endl;
}

void detectEdge(Mat& in, Mat& out) {

  blur(in, out, Size(3, 3));  

  Canny(out, out, 80, 240, KERNEL_SIZE);
}

void extractPointsFromImages(Mat inputImage, vector<int>& x, vector<int>& y){

	for(int row = 0; row < inputImage.rows; ++row){
		for (int col = 0; col < inputImage.cols; ++col)
		{
			if((int)inputImage.at<uchar>(row, col)!=0){
				x.push_back(row);
				y.push_back(col);
			}
		}
	}

	//cout << "Points in image extracted successfully.";
}

void printVectors(const vector<int>& x, const vector<int>& y){

	cout << "\nNumber of elements in X: " << x.size();
	cout << "\nNumber of elements in Y: " << y.size();

	cout << "\n[X, Y] co-ordinates are: "; 
    for (auto iterX = x.begin(), iterY = y.begin(); iterX != x.end() && iterY != y.end(); iterX++, iterY++) 
        cout << "[" << *iterX << ", " << *iterY << "] ";

    cout << endl;
}

void createCoeffMat(int n, Mat& coeffMat, const vector<int>& x, const vector<int>& y){
	
	float sumX=0, sumX2=0, sumX3=0, sumX4=0;

	for(auto iter = x.begin(); iter != x.end(); ++iter){
	
		float val = (float)*iter;
		sumX += val;
		sumX2 += pow(val, 2);
		sumX3 += pow(val, 3);
		sumX4 += pow(val, 4);
	
	}

	coeffMat.at<float>(0,0) = (float)n;
	coeffMat.at<float>(0,1) = sumX;
	coeffMat.at<float>(0,2) = sumX2;
	coeffMat.at<float>(1,0) = sumX;
	coeffMat.at<float>(1,1) = sumX2;
	coeffMat.at<float>(1,2) = sumX3;
	coeffMat.at<float>(2,0) = sumX2;
	coeffMat.at<float>(2,1) = sumX3;
	coeffMat.at<float>(2,2) = sumX4;

}

void createTargetMat(Mat& targetMat, const vector<int>& x, const vector<int>& y){

	float sumY=0, sumXY=0, sumX2Y=0;

	for (auto iterX = x.begin(), iterY = y.begin(); iterX != x.end() && iterY != y.end(); iterX++, iterY++) {
        
        float xVal = (float)*iterX, yVal = (float)*iterY;
        sumY += yVal;
        sumXY += xVal * yVal;
        sumX2Y += pow(xVal,2) * yVal; 

	}

	targetMat.at<float>(0,0) = sumY;
	targetMat.at<float>(1,0) = sumXY;
	targetMat.at<float>(2,0) = sumX2Y;

}
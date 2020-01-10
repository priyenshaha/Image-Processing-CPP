/*********************************************************************
 *Template Matching.
 *********************************************************************/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>


cv::Mat readImg(std::string src);
void showImg(std::string, cv::Mat);
int noramlisedCC(const cv::Mat, cv::Mat & , cv::Point & , cv::Point & );



int main(int argc, char ** argv) {
  //Load video and template image  s
  cv::Mat templateImage = readImg(argv[1]);
  cv::VideoCapture cap(argv[2]);

  //Conversion of template into grayscale image 
  cv::Mat grayTemp;
  cv::cvtColor(templateImage, grayTemp, CV_BGR2GRAY );

  if (argc < 3) {
    printf("USAGE: ./tempMatchExec [template] [src image] \n");
    return EXIT_FAILURE;
  }

  //Create a csv file
  std::ofstream myfile;
  myfile.open("sprint3_data_for_template_matching.csv");
  myfile << "Frame no. , Object ID, Object Info ((x1 y1) (x2 y2)),State,\n";

  // Default resolution of the frame is obtained.The default resolution is system dependent. 
  int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  //Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
  cv::VideoWriter video("result_NCC.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));
  
  //Get Framerate of video
  double fps = cap.get(CV_CAP_PROP_FPS);
  std::cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << "\n";

  //Process 100 Frames from video
  for (int i = 1; i <= 100; i++) {
    cv::Mat frame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    cv::Point pt1, pt2;
    int detect = noramlisedCC(grayTemp, frame, pt1, pt2);

    if (detect == 1)
      myfile << cv::format("%d,Car,((%d %d) (%d %d)),Detected,\n", i, pt1.x, pt1.y, pt2.x, pt2.y);
    else
      myfile << cv::format("%d,Car, - ,Not Detected,\n", i);
    std::cout << i << "\n";

    video.write(frame);
    showImg("OutputFrame", frame);
    showImg("Template", templateImage);

    // Press  ESC on keyboard to exit
    char c = (char) cv::waitKey(25);
    if (c == 27)
      break;
  }
  // When everything done, release the video capture object
  cap.release();
  video.release();
  myfile.close();

  cv::waitKey(0); //hold windows open until user presses a key
  return 0;
}



cv::Mat readImg(std::string src) {
  cv::Mat img = cv::imread(src, CV_LOAD_IMAGE_COLOR);
  return (img);
}


void showImg(std::string window, cv::Mat img) {
  cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
  cv::imshow(window, img);
}



//Template Match
//Function for normalised cross correlation
int noramlisedCC(const cv::Mat T, cv::Mat & image, cv::Point & p1, cv::Point & p2) {
  int state = 0;

  //Get dimension of template
  int height_t = T.rows;
  int width_t = T.cols;

  //Get dimension of image
  int height_i = image.rows;
  int width_i = image.cols;

  //Calculate dimension of NCC matrix
  int height_r = height_i - height_t + 1;
  int width_r = width_i - width_t + 1;


  //Create NCC result matrix
  cv::Mat result(height_r, width_r, CV_32FC1);


  cv::Mat I;
  cvtColor(image, I, CV_BGR2GRAY);


  //Performing Normalised Cross Correlation
  for (int row = 0; row < height_r; row++) {
    for (int col = 0; col < width_r; col++) {
      double sumN = 0;
      float sumD1 = 0;
      float sumD2 = 0;
      for (int rowT = 0; rowT < height_t; rowT++) {
        for (int colT = 0; colT < width_t; colT++) {
          sumN = sumN + ((T.at < uint8_t > (rowT, colT)) * (I.at < uint8_t > (row + rowT, col + colT)));
          sumD1 = sumD1 + ((T.at < uint8_t > (rowT, colT)) * (T.at < uint8_t > (rowT, colT)));
          sumD2 = sumD2 + ((I.at < uint8_t > (row + rowT, col + colT)) * (I.at < uint8_t > (row + rowT, col + colT)));
        }
      }
      double sumD = sqrt(sumD1 * sumD2);

      float score = float(sumN / sumD);
      //std::cout << "Score at (" << col <<"," << row << ") is " << score << "\n"; 
      result.at < float > (row, col) = score;
    }
  }

  double max = 0.0;
  for (int row = 0; row < height_r; row++) {
    for (int col = 0; col < width_r; col++) {
      if (result.at < float > (row, col) > max) {
        max = result.at < float > (row, col);
        p1.x = col;
        p1.y = row;
      }
    }
  }
  p2.x = p1.x + width_t;
  p2.y = p1.y + height_t;

  if (result.at < float > (p1.y, p1.x) > 0.908800) {
    cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2, 8);
    cv::putText(image,"Detected",cv::Point(p1.x-(width_t),p1.y-(height_t/2)),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,8,false);
    state = 1;
  }
  return (state);
}


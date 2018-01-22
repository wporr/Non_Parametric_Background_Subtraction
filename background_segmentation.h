#ifndef BACKSEG_H
#define BACKSEG_H
#include "imgproc.hpp"
#include "highgui.hpp"
#include "histogram.h"
#include "histogram.cpp"
#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>

using namespace cv;
using namespace std;

class background_segmentation {

private:
	//Video file path and write path
	string vid_path;
	string write_path;
	//Info about video feed
	int rows;
	int cols;
	int channels;
	int total_p;
	//Counter variables
	int frame_num;
	//Array of pixel histograms
	Hist* histarr;
	//Number of cpu threads used
	int threads;
	//Probability threashold
	double threashold = .006;
	//Histogram update interval in terms of frames
	int update_interval = 8;


public:
	background_segmentation(string, string, int);
	void run_video();
	string find_file_name(int, char = 'r');
	bool process_frame(Mat, int);
	void display_and_write(Mat, Mat, int);
	Mat smoothImg(Mat);
	void light_change(Mat&);
	void detect_ghost(Mat, Mat&);


};

#endif


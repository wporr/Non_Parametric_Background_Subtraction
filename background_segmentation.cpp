#include "background_segmentation.h"
#include "imgproc.hpp"
#include "highgui.hpp"
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;

background_segmentation::background_segmentation(string vid_path_i,
						 						 string write_path_i,
                                                 int threads_i){

	vid_path = vid_path_i;
	write_path = write_path_i;
	string first_name = find_file_name(0);
	Mat frame = imread(vid_path + first_name);

	if (frame.empty()){
		cout << "Video did not open";
		return;
	}

	frame_num = 0;
	rows = frame.rows;
	cols = frame.cols;
	channels = frame.channels();
	total_p = rows * cols;
	//Dynamically allocate the number of histograms needed
	histarr = new Hist[total_p * channels];
	histarr[0].initialize_table();	//initialize static lookup table
	//Set number of threads for OpenMP
	threads = threads_i;
	omp_set_num_threads(threads);
}

void background_segmentation::run_video(){

	int end = true;
	do {
		#pragma omp parallel
		{
			int curr_frame_num = frame_num + omp_get_thread_num();
			string file_name = find_file_name(curr_frame_num);
			Mat frame = imread(vid_path + file_name);

			end = process_frame(frame, curr_frame_num);
		}
		frame_num += threads;
	} while (end == false);
	//Delete allocated data
	delete histarr;

	return;
}

//This function is specifically for finding the file name for a video frame
// from the Changedetection.net datasets
string background_segmentation::find_file_name(int curr_frame_num,
					       					   char mode){

	string file_name = (mode == 'r') ? "in" : "out";
	string path = (mode == 'r') ? vid_path : write_path;
	int o_num = 6 - to_string(curr_frame_num + 1).length();

	if (path.find_first_of('/') == string::npos)
		file_name = (path.back() == '\\') ? file_name : "\\" + file_name;
	else
		file_name = (path.back() == '/') ? file_name : "/" + file_name;

	for (int i = 0; i < o_num; i++)
		file_name += '0';

	file_name += to_string(curr_frame_num + 1) + ".jpg";

	return file_name;
}

//Main function that processes the input frame
bool background_segmentation::process_frame(Mat frame,
					   						int curr_frame_num){

	if (frame.empty()) // exit if we've reached the end
		return true;

	Mat binary = Mat::zeros(rows, cols, CV_8UC1);
	int fornum = 1;	// counter for light detection function


	uchar* frameptr = frame.ptr<uchar>(0);		// the mat objects are continuous, so we only need
	uchar* bwptr = binary.ptr<uchar>(0);	// a pointer to the first row, saving some computation

	//Main processing loop which sorts through each frame pixel
	for (int i = 0; i < (total_p * 3); i += 3){
		if (curr_frame_num > 0){
			double probability = histarr[i].getBinVal(frameptr[i]) *
				histarr[i + 1].getBinVal(frameptr[i + 1]) *	// weights associated with the three channels
				histarr[i + 2].getBinVal(frameptr[i + 2]);	// and used to convert to binary

			if (probability < threashold){ //current probability threashold set as a parameter
				//If the probability of being a background pixel is below the threashold,
				// then we set the pixel as a foreground pixel
				bwptr[i / 3] = 255;
				fornum++;
			}

		}
		//Update the histograms
		if (curr_frame_num % update_interval == 0){
			histarr[i].updateHist(frameptr[i]);
			histarr[i + 1].updateHist(frameptr[i + 1]);
			histarr[i + 2].updateHist(frameptr[i + 2]);
		}
	}


	//Clear the histograms when there is too much foreground
	// (likely due to a lighting change) and start over
	if (fornum > (total_p * .7))
		light_change(frame);

	//Check for ghosts and reduce noise
	detect_ghost(binary, frame);
	blur(binary, binary, Size(3,3));
	Mat SmoothImg = smoothImg(binary);
	//Display frame and write to file
	display_and_write(frame, SmoothImg, curr_frame_num);

	return false;
}

void background_segmentation::display_and_write(Mat frame,
                                                Mat binary,
                                                int curr_frame_num){
	if (curr_frame_num > 0){
		namedWindow("Video: Input", CV_WINDOW_AUTOSIZE);
		namedWindow("Video: Segmented", CV_WINDOW_AUTOSIZE);
		imshow("Video: Input", frame);
		imshow("Video: Segmented", binary);
	}
	string file_name = find_file_name(curr_frame_num, 'w');
	imwrite(write_path + file_name + ".jpg", binary);
}

Mat background_segmentation::smoothImg(Mat frame)		// function to smooth out the image and eliminate noise
{
	int rows = frame.rows;
	int cols = frame.cols;
	int channels = frame.channels();
	Mat SmoothedImg = Mat::zeros(rows, cols, CV_8UC1);

	for (int count = 2; count < (rows - 2); count++){
		uchar* frameptr = frame.ptr<uchar>(count);
		uchar* smoothedptr = SmoothedImg.ptr<uchar>(count);

		for (int i = 2; i < ((cols * channels) - 2); i++){

			if (frameptr[i] != 0){
				int totalVal = 0;
				int avgVal = 0;

				totalVal += frameptr[i - 2] + frameptr[i - 1] + frameptr[i] + frameptr[i + 1] + frameptr[i + 2];

				uchar* ptrAbv1 = frame.ptr<uchar>(count - 2);
				totalVal += ptrAbv1[i - 2] + ptrAbv1[i - 1] + ptrAbv1[i] + ptrAbv1[i + 1] + ptrAbv1[i + 2];

				uchar* ptrAbv = frame.ptr<uchar>(count - 1);
				totalVal += ptrAbv[i - 2] + ptrAbv[i - 1] + ptrAbv[i] + ptrAbv[i + 1] + ptrAbv[i + 2];

				uchar* ptrBel = frame.ptr<uchar>(count + 1);
				totalVal += ptrBel[i - 2] + ptrBel[i - 1] + ptrBel[i] + ptrBel[i + 1] + ptrBel[i - 2];

				uchar* ptrBel1 = frame.ptr<uchar>(count + 2);
				totalVal += ptrBel1[i - 2] + ptrBel1[i - 1] + ptrBel1[i] + ptrBel1[i + 1] + ptrBel1[i - 2];

				avgVal = cvRound(static_cast<double>(totalVal) / 25);

				if (avgVal < 128)
					smoothedptr[i] = 0;

				else
					smoothedptr[i] = 255;
			}
		}
	}

	return SmoothedImg;
}

void background_segmentation::light_change(Mat& frame)
{
	uchar* frameptr = frame.ptr<uchar>(0);
	for (int count = 0; count < (frame.rows * frame.cols * 3); count++)
	{
		histarr[count].clearHist();
		histarr[count].updateHist(frameptr[count]);
	}

	return;
}

void background_segmentation::detect_ghost(Mat binary,
					   					   Mat& frame){
	int thresh = 100;
	Mat cannyOutput, blurred;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// find the silhouette of the binary image
	blur(binary, blurred, Size(3, 3));
	Canny(blurred, cannyOutput, thresh, thresh * 2, 3);
	findContours(cannyOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	uchar* frameptr = frame.ptr<uchar>(0);
	uchar* binptr = binary.ptr<uchar>(0);

	// record values near and outside the edges of each foreground
	// object, and subtract them to see if they are close to the same
	for (int c = 0; c < contours.size(); c++){

		if (contours[c].size() < 10)
			continue;
		float sum = 0;
		float avg = 0;
		int skipped = 0;

		for (int i = 0; i < contours[c].size(); i++){
			Point pt = contours[c][i];
			int xoff = 2;
			int yoff = 0;

			// currently skipping over contour pixels that are too close to the boundaries of the frame
			// detection should still work using pixels on the opposite side of the  ones skipped
			if (pt.x <= 2 || pt.x >= (frame.cols - 2)){
				continue;
				skipped++;
			}
			if (pt.y <= 2 || pt.y >= (frame.rows - 2)){
				continue;
				skipped++;
			}

			int val1 = frameptr[(pt.y * frame.cols * 3) + ((pt.x - 2) * 3)];
			val1 += frameptr[(pt.y * frame.cols * 3) + ((pt.x - 2) * 3 + 1)];
			val1 += frameptr[(pt.y * frame.cols * 3) + ((pt.x - 2) * 3 + 2)];
			int binval1 = binptr[(pt.y * frame.cols) + pt.x - 2];

			if (binptr[((pt.y + yoff) * frame.cols) + pt.x + xoff] == binval1){
				xoff = 0;
				yoff = 2;

				if (binptr[((pt.y + yoff) * frame.cols) + pt.x + xoff] == binval1){
					yoff = -2;

					if (binptr[((pt.y + yoff) * frame.cols) + pt.x + xoff] == binval1){
						skipped++;
						continue;
					}
				}
			}

			int val2 = frameptr[((pt.y + yoff) * frame.cols * 3) + ((pt.x + xoff) * 3)];
			val2 += frameptr[((pt.y + yoff) * frame.cols * 3) + (((pt.x + xoff) * 3) + 1)];
			val2 += frameptr[((pt.y + yoff) * frame.cols * 3) + (((pt.x + xoff) * 3) + 2)];

			if ((val1 - val2) < 1)
				sum -= (val1 - val2);
			else
				sum += (val1 - val2);
		}
		avg = sum / ((contours[c].size() - skipped) * 3);

		// clear all the histograms of the ghost object using a
		// ROI of the contour's extremes (highest and lowest values for x and y of the object)
		if (avg <= 5){
			Point pt = contours[c][0];
			int up = pt.y, down = pt.y, left = pt.x, right = pt.x;

			for (int i = 1; i < contours[c].size(); i++){
				Point pt = contours[c][i];

				if (pt.y > down)
					down = pt.y;
				if (pt.y < up)
					up = pt.y;
				if (pt.x > right)
					right = pt.x;
				if (pt.x < left)
					left = pt.x;
			}

			for (int count = up; count < down; count++){
				for (int i = left; i < right; i++){
					if (binptr[(count * frame.cols) + i] == 255){
						histarr[(count * frame.cols * 3) + (i * 3)].clearHist();
						histarr[(count * frame.cols * 3) + (i * 3) + 1].clearHist();
						histarr[(count * frame.cols * 3) + (i * 3) + 2].clearHist();

						histarr[(count * frame.cols * 3) + (i * 3)].
							updateHist(frameptr[(count * frame.cols * 3) + (i * 3)]);
						histarr[(count * frame.cols * 3) + (i * 3) + 1].
							updateHist(frameptr[(count * frame.cols * 3) + (i * 3) + 1]);
						histarr[(count * frame.cols * 3) + (i * 3) + 2].
							updateHist(frameptr[(count * frame.cols * 3) + (i * 3) + 2]);
					}
				}
			}
		}
	}

	return;
}

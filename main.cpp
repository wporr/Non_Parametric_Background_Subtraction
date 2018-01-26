#include "background_segmentation.h"
#include "background_segmentation.cpp"
#include "imgproc.hpp"
#include "highgui.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

struct timespec;
void show_stats(timespec, timespec, int);

int main(int argc, char* argv[]){

	if (argc != 4){
		cout << "There must be four arguments.\n"
			 << "The first is the path to the directory containing the video.\n"
			 << "The second is the path to where you would like to write the results.\n"
			 << "And the last is the number of CPU threads you would like to use\n"
			 << "when running the program.\n"
			 << "Make sure that you are only using videos provided by"
			 << " the Changedetection.net datasets, \n"
			 << "and that you do not specify more CPU threads than you have on your computer.\n";
	}
	string readfile = argv[1];
	string writefile = argv[2];
	int num_threads = stoi(string(argv[3]));

	timespec start, finish;
	clock_gettime(CLOCK_REALTIME, &start);

	background_segmentation my_segmentator(readfile, writefile, num_threads);
	my_segmentator.run_video();
	int last_frame = my_segmentator.get_frame_num();
	clock_gettime(CLOCK_REALTIME, &finish);
	show_stats(start, finish, last_frame);

	return 0;
}

void show_stats(timespec start, timespec finish, int last_frame){

	float elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000;
	float fps = last_frame / elapsed;

	cout << endl << "Framerate: " << fps << " FPS\n"
		 << "Seconds: " << elapsed << endl << endl;
}

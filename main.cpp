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


int main(int argc, char* argv[]){

	string readfile = argv[0];
	string writefile = argv[1];

	background_segmentation my_segmentator(readfile, writefile, 8);
	my_segmentator.run_video();
}

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "histogram.h"
#include "imgproc.hpp"
#include "highgui.hpp"

using namespace std;
using namespace cv;


float Hist::bins = 0;
int Hist::lookuptable[256];		// increases the fps when compared to just dividing within the vector call

Hist::Hist(int nobins){	//Dont call this constructior with any arguments, it will throw.
						// I created the default because it made changing the number of bins easier
	bins = nobins;		// while still having the best performance possible
	timesupdated = 0;

	for (int i = 0; i < bins; i++)
		hist[i] = 1;
	binfactor = (256 / bins);
}


void Hist::update_hist(uchar& intensity){
	hist[lookuptable[intensity]] += 1;
	timesupdated++;
}


float Hist::get_bin_val(uchar& intensity){
	//If there are no stored values, we cannot return anything
	if (timesupdated == 0)
		return NULL;
	//Normalize values
	return (hist[lookuptable[intensity]] / timesupdated);
}

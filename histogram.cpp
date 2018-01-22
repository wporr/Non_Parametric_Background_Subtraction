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

Hist::Hist(int nobins)	// dont call this constructior with any arguments, it will throw.
{				// we created the default because it made changing the number of bins easier
	bins = nobins;		// while still having the best performance possible
    binfactor = 256/bins;
	for (int i = 0; i < bins; i++)
		histogram[i] = 1/bins;
}

void Hist::updateHist(uchar& intensity)
{
	histogram[lookuptable[intensity]] += learningRate;

    for(int i=0; i<bins; i++)
        histogram[i] /= (1+learningRate);
}

float Hist::getBinVal(uchar& intensity)
{
	return histogram[lookuptable[intensity]];
}

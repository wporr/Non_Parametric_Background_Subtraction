#ifndef MYHISTV2_H
#define MYHISTV2_H
#include "imgproc.hpp"
#include "highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;


class Hist
{
private:
	int binfactor;
	// must change the number in the array below to the value
	// of the history variable manually, dynamic allocaiton is too slow
	static float bins;
	// bins set to 6 for now, to change it, change this and
	// the constructor's default value
	float histogram[6];
	float learningRate = 1/32;
	static int lookuptable[256];

public:
	Hist(int = 6);
	void updateHist(uchar&);
	float getBinVal(uchar&);
	void initialize_table(){
		for (int count = 0; count < 256; count++)
			lookuptable[count] = (count / binfactor);
		}
	void clearHist(){
		for (int i = 0; i < bins; i++)
			histogram[i] = (1/bins);
	}
};

#endif

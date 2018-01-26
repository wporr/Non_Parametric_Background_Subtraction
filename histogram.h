#ifndef HIST_H
#define	HIST_H
#include "imgproc.hpp"
#include "highgui.hpp"

using namespace std;
using namespace cv;


class Hist {

private:
	float binfactor;
	static float bins;
	//Bins set to 6 for now, to change it, change this and
	// the constructor's default value (better speed)
	int hist[6];
	static int lookuptable[256];
	float timesupdated;

public:
    	Hist(int = 6);
	void update_hist(uchar&);
	float get_bin_val(uchar&);
	void initialize_table(){
		for (int count = 0; count < 256; count++)
			lookuptable[count] = (count / binfactor);
	}
	void clear_hist(){
		for (int i = 0; i < bins; i++)
			hist[i] = 0;
		timesupdated = 0;
	}
};

#endif

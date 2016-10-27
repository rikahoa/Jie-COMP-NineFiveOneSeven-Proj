#include "trackingOBJ.h"
#include "kltTrackingOBJ.h"

using namespace cv9417;
using namespace cv9417::tracking;

trackingOBJ::trackingOBJ(void)
{
}


trackingOBJ::~trackingOBJ(void)
{
}


trackingOBJ* trackingOBJ::create(TRACKER_TYPE type)
{
	if(type == TRACKER_KLT){
		return new kltTrackingOBJ();
	}
	else{
		return 0;
	}
}

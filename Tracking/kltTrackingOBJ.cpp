#include "kltTrackingOBJ.h"
#include "commonCvFunctions.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::tracking;

kltTrackingOBJ::kltTrackingOBJ(void)
{
	max_corners = 80;
	quality_level = 0.1;
	min_distance = 5;
}


kltTrackingOBJ::~kltTrackingOBJ(void)
{
}


void kltTrackingOBJ::startTracking(const Mat& grayImg, vector<Point2f>& pts)
{
	Mat mask = createMask(grayImg.size(), pts);
	goodFeaturesToTrack(grayImg, corners, max_corners, quality_level, min_distance, mask);

	grayImg.copyTo(prevImg);
	object_position = pts;
	float d[] = {1,0,0,0,1,0,0,0,1};
	homographyMat = Mat(3,3,CV_32FC1,d).clone();
	track_status.clear();
}


bool kltTrackingOBJ::onTracking(const Mat& grayImg)
{
	std::vector<cv::Point2f> next_corners;
	std::vector<float> err;
	if(corners.size() == 0)
		return false;

	/* 
	 * calcOpticalFlowPyrLK (  
                gray_prev,	//前一帧灰度图  
                gray,		//当前帧灰度图  
                points[0],	//前一帧特征点位置  
                points[1],	//当前帧特征点位置  
                status,		//特征点被成功跟踪的标志  
                err);		//前一帧特征点点小区域和当前特征点小区域间的差，根据差的大小可删除那些运动变化剧烈的点  
	 *
	 */
	calcOpticalFlowPyrLK(prevImg, grayImg, corners, next_corners, track_status, err);

	int tr_num = 0;
	vector<unsigned char>::iterator status_itr = track_status.begin();
	while(status_itr != track_status.end()){
		if(*status_itr > 0)
			tr_num++;
		status_itr++;
	}

	/* 要求仿射变换，所以跟踪点的个数有要求 */
	if(tr_num < 6){
		return false;
	}
	else{
		homographyMat = findHomography(Mat(corners), Mat(next_corners),track_status,CV_RANSAC,5);

		if(countNonZero(homographyMat)==0){
			return false;
		}
		else{
			// 四个角的下一个位置。
			vector<Point2f> next_object_position = calcAffineTransformPoints(object_position, homographyMat);
			if(!checkPtInsideImage(prevImg.size(), next_object_position)){
				return false;
			}

			if(!checkRectShape(next_object_position)){
				return false;
			}

			int ret = checkInsideArea(next_corners, next_object_position, track_status);
			if(ret < 6){
				return false;
			}

			grayImg.copyTo(prevImg);
			corners = next_corners;
			object_position = next_object_position;
			

		}
	}
	return true;
}

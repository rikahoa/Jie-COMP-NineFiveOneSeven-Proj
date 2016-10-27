#ifndef __KLT_TRACKING_OBJ__
#define __KLT_TRACKING_OBJ__

#include <opencv2/core/core.hpp>
#include "trackingOBJ.h"

namespace cv9417{
	namespace tracking{

		class kltTrackingOBJ : public trackingOBJ
		{
		public:
			kltTrackingOBJ(void);
			virtual ~kltTrackingOBJ(void);

		private:
			cv::Mat prevImg;
			std::vector<cv::Point2f> corners;
			std::vector<cv::Point2f> object_position;
			std::vector<unsigned char> track_status;
			int max_corners;
			double quality_level;
			double min_distance;
			cv::Mat homographyMat;

		public:
			//! Start Tracking
			/*! 
			\param[in] grayImg first farme in gray scale
			\param[in] pts initial object position: pts[0]:Top Left, pts[1]:Bottom Left, pts[2]:Bottom Right, pts[3]:Top Right
			*/
			void startTracking(const cv::Mat& grayImg, std::vector<cv::Point2f>& pts);

			//! Continue Tracking
			/*!
			\param[in] grayImg input gray scale image
			\return false if tracking failed
			*/
			bool onTracking(const cv::Mat& grayImg);

			//! Get current obj position
			/*!
			\return Homography from previous frame
			*/
			cv::Mat& getHomographyMat(){return homographyMat;};

			// Jie: add
			std::vector<cv::Point2f> getObjectPosition(){return object_position;};
		};

	};
};
#endif

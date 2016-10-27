#ifndef __COMMON_CV_FUNCTIONS__
#define __COMMON_CV_FUNCTIONS__

#include <opencv2/core/core.hpp>
#include <fstream>

namespace cv9417{

	//! Write cv::Mat as binary
	/*!
	\param[out] ofs output file stream
	\param[in] out_mat mat to save
	*/
	void writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat);

	//! Read cv::Mat from binary
	/*!
	\param[in] ifs input file stream
	\param[out] in_mat mat to load
	*/
	void readMatBinary(std::ifstream& ifs, cv::Mat& in_mat);

	cv::Mat transPointVecToMat(std::vector<cv::Point2f>& pt_vec, std::vector<unsigned char>& mask);
	cv::Mat transPointVecToMat(std::vector<cv::Point2f>& pt_vec);
	cv::Mat transPointVecToMatHom(std::vector<cv::Point2f>& pt_vec);
	cv::Mat transPointVecToMat2D(std::vector<cv::Point2f>& pt_vec, std::vector<unsigned char>& mask);
	cv::Mat transPointVecToMat2D(std::vector<cv::Point2f>& pt_vec);
	std::vector<cv::Point2f> calcAffineTransformRect(cv::Size& regimg_size, cv::Mat& transMat);
	std::vector<cv::Point2f> calcAffineTransformPoints(std::vector<cv::Point2f>& pts_vec, cv::Mat& transMat);

	bool checkRectShape(std::vector<cv::Point2f>& rect_pt);
	cv::Mat createMask(cv::Size img_size, std::vector<cv::Point2f>& pts);
	int checkInsideArea(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& corner_pts, std::vector<unsigned char>& status);
	bool checkPtInsideImage(cv::Size img_size, std::vector<cv::Point2f>& pts);


	void resizeMatChannel(cv::Mat& src_mat, cv::Mat& dest_mat, double val = 0);
	template<typename _Tp> void resizeMatChannelType(cv::Mat& src_mat, cv::Mat& dest_mat, double val = 0);
	void setChannelValue(cv::Mat& dest_mat, int channel, double val = 0);
	template <typename _Tp> void setChannelValueType(cv::Mat& dest_mat, int channel, double val = 0);

	std::vector<cv::Point2f> scalePoints(std::vector<cv::Point2f>& point_vec, double scale);
};
#endif
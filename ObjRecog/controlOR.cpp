#include "controlOR.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <sstream>
#include <iostream>
#include "windows.h"

using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::or;
//#include <iostream>

controlOR::controlOR(void)
{
	detectorType = "SURF";
	descriptorType = "SURF";
	feature_detector = 0;
	descriptor_extractor = 0;
	initializeFeatureDetector();
	voteNum = 1;
	visual_words.setVoteNum(voteNum);
	image_db.setVoteNum(voteNum);
}

controlOR::~controlOR(void)
{
	releaseFeatureDetector();
}

/////// create visual word ///////
int controlOR::addFeaturesForVW(const Mat& src_img)
{
	vector<KeyPoint> kp_vec;
//	vector<float> desc_vec;
	cv::Mat desc_vec;
	
	//Step 01: 获得 “Key Points” 和 “描述子” -- 调用的都是cv lib. :-P
	extractFeatures(src_img, kp_vec, desc_vec);
	
	//Step 02: 添加 提取后的”描述子“ -- 实质为：descriptor_mat	cher.add
	//         New descriptors are added to existing train descriptors.
	visual_words.addFeatures(desc_vec);

//	kp_vec.clear();
//	desc_vec.clear();

	return 0;
}


int controlOR::createVisualWords(int cluster_number)
{
	visual_words.createVW(cluster_number);

	return 0;
}


int controlOR::registImage(const cv::Mat& src_img, int img_id)
{
	vector<KeyPoint> kp_vec;
	cv::Mat desc_vec;

	if(img_id <=0)	return -1;

	try{
		extractFeatures(src_img, kp_vec, desc_vec);

		// 01. 获得有效标签id_list，就是它自己本身。
		vector<int> id_list;
		getFeatureIdVec(desc_vec, id_list);
		
		// 02. 构造feature --> kpt的索引
		int ret = image_db.registImageFeatures(img_id, src_img.size(), kp_vec, id_list);

		if(ret < 0)
			return -1;
	}
	catch(cv::Exception e){
		throw e;
	}
	catch(std::exception e2){
		throw e2;
	}

	return 0;
}


int controlOR::removeImage(int img_id)
{
	return image_db.removeImageId(img_id);
}


void controlOR::releaseObjectDB()
{
	image_db.release();
}


vector<resultInfo> controlOR::queryImage(const Mat& src_img, int result_num)
{
	/** 04.1 **/
	vector<resultInfo>	retInfo;
	vector<KeyPoint>	kp_vec;
	cv::Mat desc_vec;

	try{
		long start = 0;
		long end = 0;

		/** 04.2 - 特征点 and 描述子 **/
		start = GetTickCount();
		extractFeatures(src_img, kp_vec, desc_vec);
		end = GetTickCount();
		cout << "<01> extract features: " << " cost: " << end-start <<" ms" << endl;

		/** 04.3 -  通过 visual word 获得 id_list，即：有效匹配的“标签” **/
		start = GetTickCount();
		vector<int> id_list;
		int ret = getFeatureIdVec(desc_vec, id_list);
		if(ret < 0)
			return retInfo;
		end = GetTickCount();
		cout << "<02> knn algorithm:    " << " cost: " << end-start <<" ms" << endl;

		/* 04.4 - 通过 visual word 判断是否参考图匹配 */
		start = GetTickCount();
		retInfo = image_db.retrieveImageId(kp_vec, id_list, src_img.size(), visual_words.getVisualWordNum(), result_num);
		end = GetTickCount();
		cout << "<03> check matrix:     " << " cost: " << end-start <<" ms" << endl;

		kp_vec.clear();
		id_list.clear();
	}
	catch(cv::Exception e)
	{
		throw e;
	}
	catch(std::exception e2)
	{
		throw e2;
	}

	return retInfo;
}


bool controlOR::setDetectorType(const std::string& detector_type)
{
	cv::Ptr<cv::FeatureDetector> tmp_detector;
	try{
		tmp_detector = FeatureDetector::create(detector_type);
		if(tmp_detector.empty()){
			return false;
		}
	}
	catch(cv::Exception e){
		return false;
	}
	this->feature_detector = tmp_detector;
	this->detectorType = detector_type;

	return true;
}


bool controlOR::setDescriptorType(const std::string& descriptor_type)
{
	cv::Ptr<cv::DescriptorExtractor> tmp_descriptor;
	try{
		tmp_descriptor = DescriptorExtractor::create(descriptor_type);
		if(tmp_descriptor.empty()){
			return false;
		}
	}
	catch(cv::Exception e){
		return false;
	}
	this->descriptor_extractor = tmp_descriptor;
	this->descriptorType = descriptor_type;

	return true;
}


//int controlOR::getFeatureIdVec(const vector<float>& desc_vec, vector<int>& id_list)
int controlOR::getFeatureIdVec(const cv::Mat& desc_vec, vector<int>& id_list)
{
	/*** 04.3.1 ***/
	if(desc_vec.empty()){
		return -1;
	}

	try{
		/*** 04.3.2 ***/
		// 获得 特征点"索引" - 有效的knn匹配的 “标签点”

		Mat indices = visual_words.querySearchDB(desc_vec);

		int size = desc_vec.rows;

		int i,j;
		for(i=0; i<size; i++)
		{
			for(j=0; j<voteNum; j++)
			{
				id_list.push_back(indices.at<int>(i,j));
			}
		}
	}
	catch(std::exception e){
		throw e;
	}

	return 0;
}


bool controlOR::saveVisualWords(const string& filename) const
{
	bool ret = visual_words.save(filename);
	return ret;
}


bool controlOR::saveVisualWordsBinary(const string& filename, const string& idxname) const
{
	bool ret = visual_words.saveBinary(filename, idxname);
	return ret;
}


bool controlOR::loadVisualWords(const string& filename)
{
	try{
		bool ret = visual_words.load(filename);
		return ret;
	}
	catch(std::exception e)
	{
//		throw e;
		return false;
	}
}


bool controlOR::loadVisualWordsBinary(const string& filename, const string& idxname)
{
	try{
		bool ret = visual_words.loadBinary(filename, idxname);
		return ret;
	}
	catch(std::exception e)
	{
//		throw e;
		return false;
	}
}


int controlOR::loadObjectDB(const string filename)
{
	try{
		/*
		 * 格式：%YAML:1.0
		 * 读取：识别obj的特征信息等。
		 */
		FileStorage cvfs(filename, FileStorage::READ);
		FileNode fn(cvfs.fs, NULL);

		// Part1: 属于controlOR
		FileNode fn1 = fn["controlOR"];
		read(fn1);

		// Part2: 属于imageDB
		FileNode fn2 = fn["imageDB"];
		image_db.read(cvfs, fn2);

		visual_words.setVoteNum(voteNum);
		image_db.setVoteNum(voteNum);
	}
	catch(std::exception e2){
		throw e2;
	}

	return 0;
}


void controlOR::read(FileNode& cvfn)
{
	voteNum = cvfn["voteNum"];
	detectorType = cvfn["detectorType"];
	descriptorType = cvfn["descriptorType"];

	feature_detector->create(detectorType);
	descriptor_extractor->create(descriptorType);
}


int controlOR::saveObjectDB(const string filename) const
{
//	image_db.save(filename.c_str());
	FileStorage cvfs(filename,FileStorage::WRITE);

	// 1. 子目录的三项内容 <-- controlOR
	write(cvfs,"controlOR");
	// 2. 
	image_db.write(cvfs, "imageDB");

	return 0;
}


void controlOR::write(FileStorage& fs, string name) const
{
	WriteStructContext ws(fs, name, CV_NODE_MAP);
	cv::write(fs, "voteNum", voteNum);
	cv::write(fs, "detectorType", detectorType);
	cv::write(fs, "descriptorType", descriptorType);
}

/////// Feature Detector ////////

// initialize
int controlOR::initializeFeatureDetector()
{
//	if(featureDetector)
	if(feature_detector || descriptor_extractor)
		releaseFeatureDetector();
//	SURF* surf_pt = new SURF(500,4,2,true);
//	featureDetector = surf_pt;
//	feature_dimention = 128;
	cv::initModule_nonfree();

	feature_detector	 = FeatureDetector::create("SURF");		
	descriptor_extractor = DescriptorExtractor::create("SURF");

	return 0;
}

int controlOR::extractFeatures(const cv::Mat& src_img, cv::vector<cv::KeyPoint>& kpt, cv::Mat& descriptor) const
{
	// extract freak

	// 调用的是cv库接口：）
	// controlOR类的生成是重点，尤其是初始化。

	/* 
	* In:  src_img
	* Out: kpt, key points of src_img
	*/
	feature_detector->detect(src_img, kpt);
	/* 
	* In:  src_img
	* In:  kpt, key points of src_img
	* Out: descriptor, descriptor of src_img.
	*/
	descriptor_extractor->compute(src_img, kpt, descriptor);



	return 0;
}

int controlOR::releaseFeatureDetector()
{
//	delete (SURF*)featureDetector;
//	featureDetector = 0;
	feature_detector.release();
	feature_detector = 0;
	descriptor_extractor.release();
	descriptor_extractor = 0;

	return 0;
}


void controlOR::setRecogThreshold(float th)
{
	image_db.setThreshold(th);
}


float controlOR::getRecogThreshold() const
{
	return	image_db.getThreshold();
}

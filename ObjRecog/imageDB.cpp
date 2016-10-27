#include "imageDB.h"
#include "commonCvFunctions.h"
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::or;

imageDB::imageDB(void)
{
	imageNum = 0;
	featureNum = 0;
//	visual_word_num = 0;
	threshold = (float)0.9;
	geo_threshold = 0.95;
	voteNum = 1;
	dist_diff_threshold = 0.005;
//	angle_diff_threshold = 0.5;
//	scale_diff_threshold = 2;
}

imageDB::~imageDB(void)
{
	release();
}

void imageDB::clearVoteTables()
{
	vector<featureVote>* vote_vec;
	map<int,vector<featureVote>*>::iterator itr = imgVote_map.begin();
	while(itr != imgVote_map.end())
	{
		vote_vec = itr->second;
		vote_vec->clear();
		itr++;
	}
}

void imageDB::releaseImgVoteMap()
{
	vector<featureVote>* vote_vec;
	map<int,vector<featureVote>*>::iterator itr = imgVote_map.begin();
	while(itr != imgVote_map.end()){
		vote_vec = itr->second;
		vote_vec->clear();
		delete vote_vec;
		itr++;
	}

	imgVote_map.clear();
}

void imageDB::release()
{
	imageNum = 0;
	featureNum = 0;
	releaseImgVoteMap();
	imgInfo_map.clear();
	keypoint_map.clear();
	feature_KPT_map.clear();
}


int imageDB::registImageFeatures(int img_id, Size img_size, vector<KeyPoint> kp_vec, vector<int> id_list)
{
	/* ToDo:  */

	//---------------------------------------------------------------------------
	imageInfo	img_info;
	img_info.feature_num = kp_vec.size();
	img_info.img_size	 = img_size;

	pair<map<int,imageInfo>::iterator,bool>	ret_insert;
	ret_insert = imgInfo_map.insert(pair< int, imageInfo >(img_id, img_info));
	//---------------------------------------------------------------------------
	// ret_insert: 
	//---------------------------------------------------------------------------

	/* ToDo:  */
	if(!(bool)(ret_insert.second)){
		return -1;
	}

	vector<featureVote>* voteTable = new vector<featureVote>;
	imgVote_map.insert(pair< int, vector<featureVote>* >(img_id, voteTable));

	featureInfo	feat_info;
	feat_info.img_id = img_id;

	int keypoint_id;
	int i;
	int size = kp_vec.size();

	// Regist image features as img_id
	// ToDo: 例外

	for(i=0; i<size; i++)		// size = 3000
	{
		if(id_list[i*voteNum] >= 0)
		{
			keypoint_id			  = getVacantKptId();
			feat_info.keypoint_id = keypoint_id;

			this->keypoint_map.insert(   pair<int,KeyPoint>(   keypoint_id,        kp_vec[i]));
			this->feature_KPT_map.insert(pair<int,featureInfo>(id_list[i*voteNum], feat_info));
			/* -------------------------------------------------
			 * feature id --> keypoint id --> keypoint detail. 
			 * -------------------------------------------------
			 */
		}
	}

	imageNum++;

	return 0;
}


// keypoint_id的分配
int imageDB::getVacantKptId()
{
	int size = keypoint_map.size();

	if(featureNum == size)
	{
		featureNum++;
		return featureNum;
	}
	else if(featureNum > size)
	{
		for(int i=1; i<=featureNum; i++)
		{
			if(keypoint_map.count(i)==0)
			{
				return i;
			}
		}
	}
}


int imageDB::removeImageId(int img_id)
{
	/////// erase img_id from imginfo_map //////
	map<int, imageInfo>::iterator imginfo_itr;
	imginfo_itr = imgInfo_map.find(img_id);

	if(imginfo_itr == imgInfo_map.end()){
		return -1;
	}

	imgInfo_map.erase(imginfo_itr);

	/////// erase img_id from feature_KPT_map and obtain the list of keypoint_id //////
	multimap<int,featureInfo>::iterator begin_itr, end_itr;
	multimap<int,featureInfo>::iterator feat_itr = feature_KPT_map.begin();
	featureInfo feat_info;
	vector<int> kpt_list;

	while(feat_itr != feature_KPT_map.end())
	{
		feat_info = feat_itr->second;
		if(feat_info.img_id == img_id){
			begin_itr = feat_itr;
			do{
				kpt_list.push_back(feat_info.keypoint_id);
				feat_itr++;
				if(feat_itr != feature_KPT_map.end()){
					feat_info = feat_itr->second;
				}
				else{
					break;
				}
			}while(feat_info.img_id == img_id);
			end_itr = feat_itr;
			feature_KPT_map.erase(begin_itr, end_itr);
			feat_itr = feature_KPT_map.begin();
		}
		else{
			feat_itr++;
		}
	}

	/////// erase keypoint_id from keypoint_map //////
	map<int, KeyPoint>::iterator kpt_itr;
	vector<int>::iterator vec_itr = kpt_list.begin();

	while(vec_itr!=kpt_list.end()){
		kpt_itr = keypoint_map.find(*vec_itr);
		keypoint_map.erase(kpt_itr);
		vec_itr++;
	}


	/////// erase img_id from imgVote_map //////
	map<int,std::vector<featureVote>*>::iterator votemap_itr;

	votemap_itr = imgVote_map.find(img_id);
	vector<featureVote>* fv_vec = votemap_itr->second;
	imgVote_map.erase(votemap_itr);

	fv_vec->clear();
	delete fv_vec;

	imageNum--;

	return img_id;
}

bool greaterResultProb(const resultInfo& result_A, const resultInfo& result_B)
{
	return result_A.probability > result_B.probability;
}

bool greaterResultMatch(const resultInfo& result_A, const resultInfo& result_B)
{
	return result_A.matched_num > result_B.matched_num;
}

vector<resultInfo> imageDB::retrieveImageId(const vector<KeyPoint>& kp_vec, const vector<int>& id_list, Size img_size, int visual_word_num, int result_num)
{
	vector<resultInfo> result_vec;
	resultInfo	result_info;

	try{
		/*
		 * 04.4.1. 构造了img_Vote_map: 存储与 “识别图 A1” 的 match_number个匹配的特征点
		 */
		voteInputFeatures(kp_vec, id_list);

		/*
		 * 04.4.2. 确定匹配成功 - 只有一项
		 */
		vector<resultInfo> tmp_result = calcMatchCountResult(kp_vec, visual_word_num);

		/*
		 * 04.4.3. 匹配成功，但不一定是个有效的凸四边形
		 */
		result_vec = calcGeometryConsistentResult(kp_vec, tmp_result, img_size, result_num);
		clearVoteTables();
	}
	catch(std::exception e2){
		throw e2;
	}

	return result_vec;
}


imageInfo imageDB::getImageInfo(int img_id)
{
	return imgInfo_map[img_id];
}


////////////// Internal Functions ///////////////////

void imageDB::voteInputFeatures(const vector<KeyPoint>& kp_vec, const vector<int>& id_list)
{
	featureInfo feat_info;
	featureVote	feat_vote;


	// featureInfo: keypoint_id; img_id;
	multimap<int,featureInfo>::iterator		 itr;

	// featureVote: in_feat_i; keypoint_id;
	map<int, vector<featureVote>*>::iterator vote_itr;

	/////////////////////////////////////////////////////////////////////////////////

	int i,j;
	int size = kp_vec.size();
	int m;

	int debug_validNum = 0;

	try{
		m=0;
		for(i=0;i<size;i++)
		{
			for(j=0;j<voteNum;j++)
			{

				if(id_list[m] >= 0)	// 非-1，即：“有效”的匹配标签
				{
					debug_validNum++;

					itr = feature_KPT_map.find(id_list[m]);		//“有效”的featureInfo

					// debug: 
					//if ((itr != feature_KPT_map.end()))
					//{
					//	cout << "nothing..." << endl;
					//}
					//else
					//{
					//	cout << "maybe..." << endl;
					//}

					//if ((itr != feature_KPT_map.end()) && (itr->first != id_list[m]))
					//{
					//	cout << "mustn't be..." << endl;
					//}


					while(itr != feature_KPT_map.end() && itr->first == id_list[m])
					{
						feat_info				= itr->second;

						/* vote */
						feat_vote.in_feat_i		= i;
						feat_vote.keypoint_id	= feat_info.keypoint_id;

						vote_itr				= imgVote_map.find(feat_info.img_id);

						if(vote_itr != imgVote_map.end())
						{
							(vote_itr->second)->push_back(feat_vote);
						}

						//if (debug_validNum == vote_itr->second->size())
						//{
						//	cout << "nothing..." << endl;
						//}
						//else
						//{
						//	cout << "wrong..." << endl;
						//}

						itr++;
					}
				}
				m++;
			}
		}

		// Debug: 
		//int total_len = vote_itr->second->size();
		//if (debug_validNum != total_len)
		//{
		//	cout << "nothing..." << endl;
		//}
	}
	catch(std::exception e)
	{
		throw e;
	}
}



// transfer input Keypoint vector and vote_table to the pair of point
// reg_vec: feature point coordinates of registered image
// query_vec: feature point coordinates of query image
void imageDB::calcPointPair(const vector<KeyPoint>& kp_vec, vector<featureVote>& vote_table, vector<Point2f>& query_pts, vector<Point2f>& reg_pts)
{
	vector<featureVote>::iterator itr = vote_table.begin();

	while(itr!=vote_table.end())
	{
		query_pts.push_back(kp_vec[itr->in_feat_i].pt);
		reg_pts.push_back(keypoint_map[itr->keypoint_id].pt);
		itr++;
	}
}


vector<resultInfo> imageDB::calcMatchCountResult(const vector<KeyPoint>& kp_vec, int visual_word_num)
{
	vector<resultInfo> result_vec;
	int img_id, match_num, reg_feats_num;
	float Pp;
	float prob;

	resultInfo	result_info;

	int in_feats_num = kp_vec.size();

	// 最终与 img A1 匹配的 几个 matching 的特征点
	vector<featureVote>* vote_table;

	try{
		// First, calculate probability from number of matching feature points
		map<int, vector<featureVote>*>::iterator vote_itr = imgVote_map.begin();

		while(vote_itr!=imgVote_map.end())
		{
			vote_table = vote_itr->second;
			match_num  = vote_table->size();

			if(match_num >= 5)
			{
				img_id = vote_itr->first;
				reg_feats_num = (imgInfo_map[img_id]).feature_num;

				// img A1 在词袋中的百分比
				Pp = (float)voteNum*reg_feats_num / visual_word_num;
				if(Pp>1)
				{
					Pp=1;
				}

				// 二项分布的概率：p = img1的词袋比重；frame特征点228个中 match_num=22 个是匹配标签的。
				// 
				prob = calcIntegBinDistribution(in_feats_num, match_num, Pp);

				if(prob >= threshold){
					result_info.img_id		= img_id;
					result_info.matched_num = match_num;
					result_info.img_size	= imgInfo_map[img_id].img_size;
					result_info.probability = prob;
					result_vec.push_back(result_info);
				}
			}

			vote_itr++;
		}
		int s = result_vec.size();
		if(s>1){
			std::sort(result_vec.begin(), result_vec.end(), greaterResultProb);
		}
		return result_vec;
	}
	catch(std::exception e2){
		throw e2;
	}
}



vector<resultInfo> imageDB::calcGeometryConsistentResult(const vector<KeyPoint>& kp_vec, const vector<resultInfo>& tmp_result_vec, Size img_size, int result_num)
{
	vector<resultInfo> result_vec;

	int s = tmp_result_vec.size();
	if(!s)
		return result_vec;

	int img_id, match_num;
	resultInfo	result_info;
	int in_feats_num = kp_vec.size();

//	int inlier_num;

	vector<featureVote>* vote_table;

	vector<Point2f> query_vec;
	vector<Point2f> reg_vec;

	// for scale & direction consistency
//	vector<KeyPoint> query_kpt;
//	vector<KeyPoint> reg_kpt;
	///////////////////////

	try{
		// Second, calculate geometrical consistency

		int count = 0;
//		float Pp, prob;
		bool shape_valid;
		int th_dist = (int)(sqrt(dist_diff_threshold * img_size.width * img_size.height / M_PI) + 0.5);

		for(int i=0;i<s && count<result_num;i++)
		{
			// 
			// 有两个R image的话，可能result_num=2, 2个result_info.
			//
			result_info = tmp_result_vec[i];
			img_id = result_info.img_id;
			match_num = result_info.matched_num;

			vote_table = imgVote_map[img_id];

			// 01 - 获得匹配点 frame vs img R1
			calcPointPair(kp_vec, *vote_table, query_vec, reg_vec);

			// 02 - 获得 Homography Matrix
			Mat poseMat = findHomography(Mat(reg_vec), Mat(query_vec), CV_RANSAC, th_dist);

			// 03 - 得到转换后的四个Corners
			vector<Point2f> pos_points = calcAffineTransformRect(imgInfo_map[img_id].img_size, poseMat);

			// 04 - 转换后的四个点的有效性，计算向量的差乘
 			shape_valid = checkRectShape(pos_points);

			reg_vec.clear();
			query_vec.clear();

			if(shape_valid){
				result_info.pose_mat		= poseMat;
				result_info.object_position = pos_points;

				result_vec.push_back(result_info);
				count++;
			}
		}
		/* 没匹配成功的话，返回NULL */
		return result_vec;
	}
	catch(std::exception e2){
		throw e2;
	}
}


// position
int imageDB::countAffineInlier(vector<Point2f> &src_pts, vector<Point2f> &dest_pts, Mat &affMat, double dist_threshold)
{
	try{
		Mat src_mat = transPointVecToMatHom(src_pts);

		Mat calc_mat = affMat * src_mat;
		double sub_x, sub_y, dist;
		int count = 0;

		int size = src_pts.size();
		for(int i=0; i<size; i++){
			sub_x = dest_pts[i].x - calc_mat.at<double>(0,i) / calc_mat.at<double>(2,i);
			sub_y = dest_pts[i].y - calc_mat.at<double>(1,i) / calc_mat.at<double>(2,i);
			dist = sqrt(sub_x * sub_x + sub_y * sub_y);
			if(dist < dist_threshold)
				count++;
		}
		return count;
	}
	catch(std::exception e2){
		throw e2;
	}
}


float imageDB::calcIntegBinDistribution(int in_feats_num, int match_num, float Pp)
{
	float prob = 0;
//	float Np = 1.0 - Pp;
	float tmp1;
	float logPp = log(Pp);
	float logNp = log((float)1.0 - Pp);
	int i,j;

	for(i=0;i<=match_num;i++){
		tmp1 = 0;
		for(j=0;j<i;j++){
			tmp1 += (float)log((double)(in_feats_num - j));
			tmp1 -= (float)log((double)(j+1));
		}
		tmp1 += logPp*i;
		tmp1 += logNp*(in_feats_num-i);
		prob += exp(tmp1);
		if(prob > 1){
			prob = 1;
			break;
		}
	}

	return prob;
}


////////////////////////////////////////////

void imageDB::setThreshold(float th)
{
	this->threshold = th;
}


float imageDB::getThreshold() const
{
	return this->threshold;
}


void imageDB::setVoteNum(int vote_num)
{
	voteNum = vote_num;
}

///////////// Load & Save ////////////////////
int imageDB::save(const string& filename) const
{
	FileStorage cvfs(filename,CV_STORAGE_WRITE);
	this->write(cvfs, "imageDB");

	return 0;
}


int imageDB::load(const string& filename)
{
	try{
		FileStorage cvfs(filename,CV_STORAGE_READ);
		FileNode cvfn(cvfs.fs, NULL);
		FileNode cvfn2 = cvfn["imageDB"];
		this->read(cvfs, cvfn2);
	}
	catch(std::exception e2){
		throw e2;
	}

	return 0;
}


int imageDB::write(FileStorage& cvfs, const string& name) const
{
	WriteStructContext ws(cvfs, name, CV_NODE_MAP);

	cv::write(cvfs,"imageNum",imageNum);
	cv::write(cvfs,"featureNum",featureNum);	// 两张图片的feature共有6000个。
	cv::write(cvfs,"threshold",threshold);
	cv::write(cvfs,"geo_threshold",geo_threshold);
	cv::write(cvfs,"dist_diff_threshold",dist_diff_threshold);
	writeFeatureKptMap(cvfs, "feature_KPT_map");
	writeKeyMap(cvfs, "keypoint_map");
	writeImgInfoMap(cvfs, "imgInfo_map");

	return 0;
}


int imageDB::read(const FileStorage& cvfs, const FileNode& node)
{
	// 关键是解析db.txt文件

	FileNode fn;
	imageNum = node["imageNum"];
	featureNum = node["featureNum"];
	threshold = node["threshold"];
	geo_threshold = node["geo_threshold"];
	dist_diff_threshold = node["dist_diff_threshold"];
	fn = node["feature_KPT_map"];
	readFeatureKptMap(cvfs, fn);
	fn = node["keypoint_map"];
	readKeyMap(cvfs,fn);
	fn = node["imgInfo_map"];
	readImgInfoMap(cvfs,fn);

	return 0;
}


int imageDB::writeFeatureKptMap(FileStorage& cvfs, const string& name) const
{
	try{
		WriteStructContext ws(cvfs, name, CV_NODE_SEQ);

		multimap<int,featureInfo>::const_iterator itr = feature_KPT_map.begin();
		featureInfo	feature_info;
		while(itr!= feature_KPT_map.end()){
			WriteStructContext ws2(cvfs, "", CV_NODE_MAP);
			cv::write(cvfs, "feature_id", itr->first);
			feature_info = itr->second;
			cv::write(cvfs, "keypoint_id",feature_info.keypoint_id);
			cv::write(cvfs, "img_id", feature_info.img_id);
			itr++;
		}
	}
	catch(std::exception e2){
		throw e2;
	}

//	cvEndWriteStruct(cvfs);
	return 0;
}

int imageDB::readFeatureKptMap(const FileStorage& cvfs, const FileNode& node)
{
	feature_KPT_map.clear();

	int feature_id;
	featureInfo	feature_info;

	FileNodeIterator	it = node.begin();
	while(it != node.end()){
		feature_id = (int)(*it)["feature_id"];
		feature_info.keypoint_id = (int)(*it)["keypoint_id"];
		feature_info.img_id = (int)(*it)["img_id"];
		feature_KPT_map.insert(pair<int, featureInfo>(feature_id,feature_info));
		it++;
	}

	return 0;
}


int imageDB::writeKeyMap(FileStorage& cvfs, const string& name) const
{
	WriteStructContext ws(cvfs, name, CV_NODE_SEQ);

	map<int,KeyPoint>::const_iterator itr = keypoint_map.begin();

	while(itr!= keypoint_map.end())
	{
		WriteStructContext ws2(cvfs, "", CV_NODE_MAP);

		cv::write(cvfs, "keypoint_id", itr->first);

		vector<KeyPoint>	kpt_vec;
		kpt_vec.push_back(itr->second);
		cv::write(cvfs, "KeyPoint", kpt_vec);
		itr++;
	}

	return 0;
}


int imageDB::readKeyMap(const FileStorage& cvfs, const FileNode& node)
{
	keypoint_map.clear();

	int keypoint_id;

	FileNodeIterator	it = node.begin();
	while(it != node.end()){
		vector<KeyPoint> keypt_vec;
		keypoint_id = (int)(*it)["keypoint_id"];
		cv::read((*it)["KeyPoint"], keypt_vec);
		keypoint_map.insert(pair<int,KeyPoint>(keypoint_id,keypt_vec[0]));
		it++;
	}

	return 0;
}


int imageDB::writeImgInfoMap(FileStorage& cvfs, const string& name) const
{
	WriteStructContext ws(cvfs, name, CV_NODE_SEQ);

	imageInfo	img_info;
	map<int, imageInfo>::const_iterator itr = imgInfo_map.begin();
	while(itr!= imgInfo_map.end()){
		WriteStructContext ws2(cvfs, "", CV_NODE_SEQ);
		cv::write(cvfs, itr->first);
		img_info = itr->second;
		cv::write(cvfs, img_info.feature_num);
		cv::write(cvfs, img_info.img_size.width);
		cv::write(cvfs, img_info.img_size.height);
		itr++;
	}
	return 0;
}


int imageDB::readImgInfoMap(const FileStorage& cvfs, const FileNode& node)
{
	imgInfo_map.clear();
	releaseImgVoteMap();

	int img_id;
	imageInfo	img_info;

	FileNodeIterator	it = node.begin();
	while(it != node.end()){
		img_id = (int)(*it)[0];
		img_info.feature_num = (int)(*it)[1];
		img_info.img_size = Size((int)(*it)[2],(int)(*it)[3]);
		imgInfo_map.insert(pair<int,imageInfo>(img_id,img_info));

		// create voteTable
		vector<featureVote>* voteTable = new vector<featureVote>;
		imgVote_map.insert(pair<int,vector<featureVote>*>(img_id, voteTable));

		it++;
	}

	return 0;
}
////////////////////////////////////////////////////////////////////////////

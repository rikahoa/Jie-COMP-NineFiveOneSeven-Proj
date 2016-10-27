#include "visualWords.h"
#include "commonCvFunctions.h"
#include <opencv2/flann/flann.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::flann;
using namespace cv9417;
using namespace cv9417::or;

visualWords::visualWords(void)
{
	//	searchDB = 0;
	this->matcherType = "FlannBased";

	this->descriptor_matcher = DescriptorMatcher::create(matcherType);

	voteNum = 1;
	radius = 0.2;
}

visualWords::~visualWords(void)
{
	release();
}

void visualWords::release()
{
	descriptor_matcher->clear();
}

bool visualWords::isReady()
{
	if(descriptor_matcher.empty() || descriptor_matcher->empty()){
		return false;
	}
	return true;
}

int visualWords::getVisualWordNum() const
{
	int num = 0;
	vector<Mat> mat_vec = descriptor_matcher->getTrainDescriptors();
	vector<Mat>::iterator itr, it_end;
	it_end = mat_vec.end();
	for(itr = mat_vec.begin(); itr != it_end; itr++){
		num += itr->rows;
	}
	//	return this->DBdata.rows;
	return num;
}

void visualWords::setVoteNum(int vote_num)
{
	voteNum = vote_num;
}

//int visualWords::addFeatures(vector<float>& feature)
void visualWords::addFeatures(const cv::Mat& feature)
{
	vector<Mat> feature_vector;
	feature_vector.push_back(feature);

	/*
	* In: feature_vector.
	* Add feature vector of each training image into flannBasedMatcher.
	*/
	descriptor_matcher->add(feature_vector);	

}

void visualWords::createVW(int cluster_num)
{
	//其实是FlannBased的描述子的API。
	try{
		// if cluster_num <= 0 then feature vector index is created without clustering. 
		//cluster_num = 100;
		if(cluster_num > 0)
		{
#if 1
			///////// kmeans clustering to create visual words ///////////////
			vector<Mat> feature_vec = descriptor_matcher->getTrainDescriptors(); // ***

			if(!feature_vec.empty() && feature_vec[0].type() == CV_32FC1)
			{
				Mat featureMat;
				convertFeatureMat(feature_vec, featureMat);

				Mat label(featureMat.rows, 1, CV_32SC1);
				Mat centroid(cluster_num, featureMat.cols,featureMat.type());

				double ret = kmeans(featureMat, cluster_num, label, TermCriteria(TermCriteria::MAX_ITER, 10, 0.5),2,KMEANS_PP_CENTERS, centroid);

				descriptor_matcher->clear();

				descriptor_matcher->add(centroid);
			}
#endif
		}

		// FlannBased的training 即构成 visual word.
		// update kd-tree.
		descriptor_matcher->train();
	}
	catch(std::exception e2){
		throw e2;
	}
}


void visualWords::convertFeatureMat(const vector<cv::Mat>& feature, cv::Mat& featureMat)
{
	//int row = feature.size() / feature_dim;
	int row = 0;
	int feature_dim = feature[0].cols;
	vector<cv::Mat>::const_iterator itr;
	for(itr=feature.begin(); itr!=feature.end(); itr++){
		row += itr->rows;
		if(feature_dim != itr->cols){

		}
	}
	featureMat.create(row, feature_dim, feature[0].type());

	unsigned char* feature_pt = featureMat.data;
	int num_bytes = featureMat.elemSize();
	int feature_size = feature.size();
	int total;
	for(int i=0; i<feature_size; i++){
		total = feature[i].total();
		memcpy(feature_pt, feature[i].data, total * num_bytes);
		feature_pt += total;
	}
}

///////// load & save /////////////
bool visualWords::saveIndex(const string& filename) const
{
	try{
		cv::FileStorage fs(filename, FileStorage::WRITE);

		//写在标签"VW_Index"下
		writeIndex(fs, "VW_Index");
		return true;
	}
	catch(std::exception e){
		//		std::cerr << e.what() << std::endl;
		return false;
	}
}


void visualWords::writeIndex(cv::FileStorage& FS, const std::string& flagName) const
{
	try{
		// 
		cv::WriteStructContext ws(FS, flagName, CV_NODE_MAP);

		cv::write(FS, "matcherType", matcherType);

		// 只是写入了matcherType: FlannBased。
		descriptor_matcher->write(FS);
	}
	catch(cv::Exception e)
	{
		throw e;
	}
}

bool visualWords::loadIndex(const string& filename)
{
	try{
		cv::FileStorage fs(filename, FileStorage::READ);
		readIndex(fs["VW_Index"]); 
		return true;
	}
	catch(std::exception e){
		//		std::cerr << e.what() << std::endl;
		return false;
	}
}


void visualWords::readIndex(const cv::FileNode& FN)
{
	try{
		cv::read(FN["matcherType"], this->matcherType, "");
		this->descriptor_matcher = DescriptorMatcher::create(matcherType);	
		descriptor_matcher->read(FN);
	}
	catch(cv::Exception e){
		throw e;
	}
}


bool visualWords::save(const string& filename) const
{
	try{
		FileStorage	FS(filename, FileStorage::WRITE);

		this->writeIndex(FS, "index");
		this->write(FS, "visualWords");
		return true;
	}
	catch(std::exception e)
	{
		return false;
	}
}


bool visualWords::saveBinary(const string& filename, const string& idx_filename) const
{
	try{
		// save index data;
		this->saveIndex(idx_filename);

		save_vw_binary(filename);
		return true;
	}
	catch(std::exception e)
	{
		return false;
	}
}


bool visualWords::load(const string& filename)
{
	try{
		FileStorage cvfs(filename, FileStorage::READ);
		this->readIndex(cvfs["index"]);
		this->read(cvfs["visualWords"]);
		return true;
	}
	catch(std::exception e2){
		return false;
	}
}


bool visualWords::loadBinary(const string& filename, const string& idx_filename)
{
	try{
		bool ret;
		ret = loadIndex(idx_filename);
		if(!ret){
			return ret;
		}

		ret = load_vw_binary(filename);
		return ret;
	}
	catch(std::exception e2){
		return false;
	}
}


bool visualWords::save_vw_binary(const string& filename) const
{
	// 利用二进制流写入config
	ofstream ofs(filename, ios::binary);
	if(!ofs.is_open()){
		//		throw new orArgException("failed to open " + filename);
		return false;
	}
	//	int type = DBdata.type();
	ofs.write((const char*)"vw", 2);
	ofs.write((const char*)(&version), sizeof(int));
	ofs.write((const char*)(&radius), sizeof(radius));
	vector<Mat> train_desc = descriptor_matcher->getTrainDescriptors();

	int num = train_desc.size();
	ofs.write((const char*)(&num), sizeof(num));
	vector<Mat>::iterator itr;
	for(itr = train_desc.begin(); itr != train_desc.end(); itr++){
		writeMatBinary(ofs, *itr);
	}

	return true;
}


bool visualWords::load_vw_binary(const string& filename)
{
	ifstream ifs(filename, ios::binary);
	if(!ifs.is_open()){
		std::cerr << "failed to open " << filename << std::endl;
		return false;
	}
	char header[2];
	ifs.read(header, sizeof(header));
	if(memcmp(header, (const void*)"vw", 2)!=0)
	{
		std::cerr << "wrong format file " << filename << std::endl;
		return false;
	}

	int ver;
	ifs.read((char*)(&ver), sizeof(int));
	if(ver != version)
	{
		std::cerr << "wrong version file: " << filename << std::endl;
		return false;
	}

	ifs.read((char*)(&radius), sizeof(radius));

	int num;
	ifs.read((char*)(&num), sizeof(num));
	descriptor_matcher->clear();
	for(int i=0; i<num; i++){
		Mat DBdata;
		readMatBinary(ifs, DBdata);
		vector<Mat> in_mat_vec;
		in_mat_vec.push_back(DBdata);
		descriptor_matcher->add(in_mat_vec);
	}

	return true;
}


void visualWords::write(FileStorage& fs, const string& name) const
{
	WriteStructContext ws(fs, name, CV_NODE_MAP);
	cv::write(fs, "version", version);
	cv::write(fs, "radius", radius);

	vector<Mat> train_desc = descriptor_matcher->getTrainDescriptors();
	WriteStructContext ws2(fs, "TrainDescriptors", CV_NODE_SEQ);
	vector<Mat>::iterator itr;
	for(itr = train_desc.begin(); itr != train_desc.end(); itr++)
	{
		cv::write(fs, std::string(), *itr);
	}
}


void visualWords::read(const FileNode& node)
{
	//	int	i;

	try{
		int ver = node["version"];
		radius = node["radius"];

		descriptor_matcher->clear();
		cv::FileNode data_node = node["TrainDescriptors"];
		cv::FileNodeIterator itr;
		cv::FileNodeIterator it_end = data_node.end();
		for(itr=data_node.begin(); itr != it_end; itr++){
			cv::Mat DBdata;
			cv::read(*itr, DBdata);

			vector<Mat> mat_vec;
			mat_vec.push_back(DBdata);
			descriptor_matcher->add(mat_vec);
		}
	}
	catch(std::exception e2){
		throw e2;
	}
}


Mat visualWords::querySearchDB(const Mat& features)
{
	int knn_size = voteNum + 1;
	int size = features.size().height;

	Mat indices(size, knn_size, CV_32SC1);
	//		Mat dists(size, knn_size, CV_32FC1);

	// search nearest descriptor in database
	vector<vector<DMatch> > match_index;
	descriptor_matcher->knnMatch(features, match_index, 4);

	int debug_validNumberOfKnn = 0;
	for(int y=0; y<size; y++)			// size = number of features
	{
		for(int x=0; x<voteNum; x++)	// voteNum = 1
		{
			DMatch d_match(match_index[y][x]);

			if(d_match.distance >= radius)
			{
				// If the descriptor vector is far from the matched centroid vector,
				// filter it out.
				indices.at<int>(y,x) = -1;
			}
			else
			{
				// Good matches will be stored in indices for next step.
				indices.at<int>(y,x) = d_match.trainIdx;
				debug_validNumberOfKnn++;
			}
		}
	}

	return indices;

}


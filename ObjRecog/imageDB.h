#ifndef __IMAGE_DB__
#define __IMAGE_DB__

#include <opencv2/features2d/features2d.hpp>

namespace cv9417{
	namespace or{

		typedef struct{
			int in_feat_i;
			int keypoint_id;
		}featureVote;

		typedef struct{
			int keypoint_id;
			int img_id;
		}featureInfo;

		typedef struct{
			int feature_num;
			cv::Size img_size;
		}imageInfo;

		typedef struct{
			int	img_id;
			float	probability;
			int	matched_num;
			cv::Mat	pose_mat;
			cv::Size img_size;
			std::vector<cv::Point2f> object_position;
		}resultInfo;

		class imageDB
		{
		public:
			imageDB(void);
			~imageDB(void);

			// Operational Functions
			int registImageFeatures(int img_id, cv::Size img_size, std::vector<cv::KeyPoint> kp_vec, std::vector<int> id_list);	// Add image feature id to DB
			std::vector<resultInfo> retrieveImageId(const std::vector<cv::KeyPoint>& kp_vec, const std::vector<int>& id_list, cv::Size img_size, const int visual_word_num, int result_num = 1);	// Add image feature id to DB
			int removeImageId(int img_id);
			void setThreshold(float th);
			float getThreshold() const;
			imageInfo getImageInfo(int img_id);
			void setVoteNum(int vote_num);
			void release();

			// Load & Save
			int load(const std::string& filename);
			int save(const std::string& filename) const;
			int read(const cv::FileStorage& cvfs, const cv::FileNode& node);
			int write(cv::FileStorage& cvfs, const std::string& name) const;

		private:
			int readFeatureKptMap(const cv::FileStorage& cvfs, const cv::FileNode& node);
			int writeFeatureKptMap(cv::FileStorage& cvfs, const std::string& name) const;
			int readKeyMap(const cv::FileStorage& cvfs, const cv::FileNode& node);
			int writeKeyMap(cv::FileStorage& cvfs, const std::string& name) const;
			int readImgInfoMap(const cv::FileStorage& cvfs, const cv::FileNode& node);
			int writeImgInfoMap(cv::FileStorage& cvfs, const std::string& name) const;

			// Internal Functions
			int getVacantKptId();
			void voteInputFeatures(const std::vector<cv::KeyPoint>& kp_vec, const std::vector<int>& id_list);
			std::vector<resultInfo> calcMatchCountResult(const std::vector<cv::KeyPoint>& kp_vec, int visual_word_num);
			std::vector<resultInfo> calcGeometryConsistentResult(const std::vector<cv::KeyPoint>& kp_vec, const std::vector<resultInfo>& tmp_result_vec, cv::Size img_size, int result_num);
			void calcPointPair(const std::vector<cv::KeyPoint>& kp_vec, std::vector<featureVote>& vote_vec, std::vector<cv::Point2f>& query_pts, std::vector<cv::Point2f>& reg_pts);
			float calcIntegBinDistribution(int in_feats_num, int match_num, float Pp);
			int countAffineInlier(std::vector<cv::Point2f>& src_pts, std::vector<cv::Point2f>& dest_pts, cv::Mat& affMat, double dist_threthold);

			void clearVoteTables();
			void releaseImgVoteMap();

		private:
			std::multimap<int,featureInfo>	feature_KPT_map;
			std::map<int,cv::KeyPoint>	keypoint_map;
			std::map<int,imageInfo>	imgInfo_map;
			std::map<int,std::vector<featureVote>*>	imgVote_map;

			//	int visual_word_num;
			int imageNum;
			int featureNum;
			int voteNum;
			float threshold;
			float geo_threshold;
			double dist_diff_threshold;
		};

	};
};
#endif
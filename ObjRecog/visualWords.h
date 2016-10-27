#ifndef __VISUAL_WORDS__
#define __VISUAL_WORDS__

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace cv9417{
	namespace or{

		class visualWords
		{
		public:
			visualWords(void);
			~visualWords(void);

			//! Release All
			void release();

			bool isReady();

			void addFeatures(const cv::Mat& feature);
			void createVW(int cluster_num = 0);

			// Load & Save
			bool save(const std::string& filename) const;
			bool saveBinary(const std::string& filename, const std::string& idx_filename) const;
			void write(cv::FileStorage& FS, const std::string& name) const;
			bool load(const std::string& filename);
			bool loadBinary(const std::string& filename, const std::string& idx_filename);
			void read(const cv::FileNode& node);

			void setVoteNum(int vote_num);

			int getVisualWordNum() const;

			cv::Mat querySearchDB(const cv::Mat& features);

		private:
			// search DB functions
			static void convertFeatureMat(const std::vector<cv::Mat>& src_feature, cv::Mat& dest_feature);
			bool save_vw_binary(const std::string& filename) const;
			bool load_vw_binary(const std::string& filename);
			bool saveIndex(const std::string& filename) const;
			bool loadIndex(const std::string& filename);
			void writeIndex(cv::FileStorage& FS, const std::string& name) const;
			void readIndex(const cv::FileNode& node);

		private:
			std::string matcherType;
			cv::Ptr<cv::DescriptorMatcher>	descriptor_matcher;
			int voteNum;
			float radius;

			static const int version = 120;
		};

	};
};
#endif
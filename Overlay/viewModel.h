#ifndef __VIEW_MODEL__
#define __VIEW_MODEL__

#include <opencv2/core/core.hpp>
#include "modelobject.h"
#include "GLMetaseq.h"
#include "modelobject.h"

namespace cv9417{

	namespace overlay{

		//----------------
		// 01. MODEL_INFO
		//----------------
		typedef struct
		{
			modelObject* model;
			double scale;
			std::string modelFilename;
			cv::Mat initRot;
			cv::Mat initTrans;
			cv::Point2f	markerCenter;
			cv::Size markerSize;

		}MODEL_INFO;

		//----------------
		// 02. arModel
		//----------------
		class arModel
		{
		private:
			static arModel* vmInstance;
			arModel(void);
			arModel(arModel*){}
			~arModel(void);

		public:
			//	arModel(void);
			//	~arModel(void);

			static arModel* getInstance(){
				if(!vmInstance){
					vmInstance = new arModel();
				}
				return vmInstance;
			}

			static void deleteInstance(){
				if(vmInstance){
					delete vmInstance;
				}
				vmInstance = 0;
			}

		public:
			cv::Mat	resized_frame;

			int two_power_width;
			int two_power_height;

			int window_width;
			int window_height;

			int capture_width;
			int capture_height;

			GLuint texture[1];
			double aspect_rate;
			float focal_length;

			cv::Mat cameraMatrix;

			std::map<int,MODEL_INFO>	model_map;

			int wait_frames;
			MODEL_INFO wait_model;

		protected:	
			cv::Mat accHomMat;
			int mat_type;
			MODEL_INFO*	curModel;
			bool mirror_f;
			cv::Mat mirrorMat;

		public:
			bool init(cv::Size& frame_size, cv::Mat& cameraMat);
			bool init(cv::Size& frame_size, cv::Mat& cameraMat, int type);
			void resize(int win_w, int win_h);
			void updateTexture(cv::Mat& frame);
			void release();
			void exitFunc();

			bool setTwoPowerSize(int w, int h);
			void setCameraMatrix(cv::Mat& cameraMat);
			void setMirrorMode(bool flag);
			void setFocalLength(float len);

			void drawScene(cv::Mat& img);
			void drawObject(cv::Mat& homographyMat, int seq_id);
			void drawWaitModel(int seq_id);
			template<typename _Tp> void drawObjectType(cv::Mat& homographyMat, int seq_id);
			void initAccHomMat();
			bool setRecogId(int id, cv::Mat& homMat);
			bool addModel(int id, cv::Size& markerSize, int model_type, const std::string& model_filename, double scale = 1.0);
			bool addModel(int id, cv::Size& markerSize, int model_type, const std::string& model_filename, double scale, cv::Mat& initRot, cv::Mat& initTrans);
			void releaseModel();

			bool addWaitModel(int wait_frame_num, int model_type, const std::string& model_filename, double scale = 1.0);
			bool addWaitModel(int wait_frame_num, int model_type, const std::string& model_filename, double scale, cv::Mat& initRot, cv::Mat& initTrans);
			void releaseWaitModel();
		};

	};
};
#endif
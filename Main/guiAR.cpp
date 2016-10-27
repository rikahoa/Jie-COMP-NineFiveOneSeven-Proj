#include <iostream>
#include <GL/glut.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include "guiAR.h"
#include "trackingOBJ.h"
#include "viewModel.h"
#include "commonCvFunctions.h"


using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::or;
using namespace cv9417::tracking;
using namespace cv9417::overlay;

controlOR	*ctrlOR  = 0;	//#
trackingOBJ	*trckOBJ = 0;	//#
arModel		*model;
VideoCapture cap(0);

string configFile_path = ".\\config.xml";



int seq_id = 0;	
int wait_seq_id = 0;
bool track_f = false;
int query_scale=1;
Mat image_buf;

namespace cv9417{

///////////////////////////////////////////////////////////////////////////////

	void setControlOR(controlOR& ctrlOR_in)
	{
		ctrlOR = &ctrlOR_in;
	}

	void setConfigFile(string& conf_xml)
	{
		configFile_path = conf_xml;
	}

///////////////////////////////////////////////////////////////////////////////

	void getObjRecImgBuf(const FileStorage cvfs, const int frame_width, const int frame_height)
	{
		int max_size = 0;
		cvfs["max_query_size"] >> max_size;


		int frame_max_size;
		if(frame_width > frame_height){
			frame_max_size = frame_width;
		}
		else{
			frame_max_size = frame_height;
		}

		query_scale = 1;
		while((frame_max_size / query_scale) > max_size){
			query_scale*=2;
		}

		image_buf.create(frame_height/query_scale, frame_width/query_scale, CV_8UC1);
	}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

	void loadVisualWords(FileStorage cvfs)
	{
		FileNode fn;
		string	 vwfile;
		string	 idxfile;

		// Config.xml中的VisualWord节点中的：
		//		visualWord	地址
		//		index		地址
		//
		// 获取了子配置文件的地址。

		fn = cvfs["VisualWord"];
		fn["visualWord"] >> vwfile;		// <-- config/visualWord.bin
		fn["index"]		 >> idxfile;	// <-- config/vw_index.txt

		if(idxfile.empty()){
			ctrlOR->loadVisualWords(vwfile);
		}
		else{
			ctrlOR->loadVisualWordsBinary(vwfile, idxfile);	
		}
	}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

	void loadObjectDB(FileStorage cvfs)
	{
		string objectDB = cvfs["ObjectDB"];
		ctrlOR->loadObjectDB(objectDB);	//*** <-- image特征点信息
	}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
	
	boolean checkPosition(std::vector<cv::Point2f> position_corners, int x, int y)
	{
		//cout << position_corners[0] << endl;
		//cout << position_corners[1] << endl;
		//cout << position_corners[2] << endl;
		//cout << position_corners[3] << endl;
		int threshold = 10;
		int dis_x = 0;
		int dis_y = 0;

		for ( int i = 0; i < 4; i++)
		{
			if (position_corners[i].x >= x)
			{
				dis_x = position_corners[i].x - x;
			}
			else
			{
				dis_x = x - position_corners[i].x;
			}

			if (position_corners[i].y >= y)
			{
				dis_y = position_corners[i].y - y;
			}
			else
			{
				dis_y = y - position_corners[i].y;
			}
		}

		return false;
	}
	


	// Jie: ...
	// using .ptr and []
	void drawLoLoModel(cv::Mat &frame, std::vector<cv::Point2f> position_corners)
	{
		int threshold = 5;

		//// Homography.
		//Mat pose_mat = recog_result[0].pose_mat.clone();

		//// Four corners.
		//std::vector<cv::Point2f> object_position = recog_result[0].object_position;

		//cout << position_corners[0] << endl;	// 左上
		//cout << position_corners[1] << endl;	// 左下
		//cout << position_corners[2] << endl;	// 右下
		//cout << position_corners[3] << endl;	// 右上


		std::vector<cv::Point2f> object_postions_draw;
		for (int i = 0; i < 4; i++)
		{
			cv::Point2f point;
			point.x = position_corners[i].x;
			point.y = position_corners[i].y;
			object_postions_draw.push_back(point);
		}

		//=========================================================================
		// 01. Draw four corners.

		int nr= frame.rows; // number of rows
		int nc= frame.cols; // total number of elements per line
		int channels= 3;

		// Row.
		for (int j=0; j<nr; j++) 
		{
			uchar* data= frame.ptr<uchar>(j);

			// Col.
			for (int i=0; i<nc; i++) 
			{
				//------------------------------------------------------------------
				for (int index_corner = 0; index_corner < 4; index_corner++)
				{
					if (abs(object_postions_draw[index_corner].x - i) < threshold &&
						abs(object_postions_draw[index_corner].y - j) < threshold)
					{
						data[i*channels]	= 0;
						data[i*channels+1]	= 0;
						data[i*channels+2]	= 200;
					}
				}
				//------------------------------------------------------------------
			}                  
		}

		//=========================================================================
		// 02. struct new key points.

		Mat transformedImage;
		Mat model_image;
		if (rand()%2 == 0)
		{
			model_image  = imread(".\\img\\M1.jpg");
		}
		else
		{
			model_image = imread(".\\img\\M2.jpg");
		}

		int g_val_zoom = 4;
		cv::resize(model_image, model_image, Size(model_image.cols / g_val_zoom, model_image.rows / g_val_zoom));

		std::vector<cv::Point2f> oriImage_corners;

		cv::Point2f point_topLeft;
		point_topLeft.x = 0;
		point_topLeft.y = 0;
		oriImage_corners.push_back(point_topLeft);

		cv::Point2f point_bottonLeft;
		point_bottonLeft.x = 0;
		point_bottonLeft.y = model_image.rows;
		oriImage_corners.push_back(point_bottonLeft);

		cv::Point2f point_bottonRight;
		point_bottonRight.x = model_image.cols;
		point_bottonRight.y = model_image.rows;
		oriImage_corners.push_back(point_bottonRight);

		cv::Point2f point_topRight;
		point_topRight.x = model_image.rows;
		point_topRight.y = 0;
		oriImage_corners.push_back(point_topRight);

		////////////////////////////////////////////////

		Mat pose_mat = findHomography((Mat)oriImage_corners, (Mat)object_postions_draw, CV_RANSAC);

		warpPerspective(model_image, transformedImage, pose_mat, Size(frame.cols, frame.rows));

		// Row.
		for (int j=0; j<nr; j++) 
		{
			uchar* data= frame.ptr<uchar>(j);
			uchar* data2= transformedImage.ptr<uchar>(j);

			// Col.
			for (int i=0; i<nc; i++) 
			{
				//------------------------------------------------------------------
				for (int index_corner = 0; index_corner < 4; index_corner++)
				{
					if (data2[i*channels] == 0 &&
						data2[i*channels+1] == 0 &&
						data2[i*channels+2] == 0)
					{
						;
					}
					else
					{
						data[i*channels]	= data2[i*channels];
						data[i*channels+1]	= data2[i*channels+1];
						data[i*channels+2]	= data2[i*channels+2];
					}
				}
				//------------------------------------------------------------------
			}                  
		}
	}


	void setARConfig(const int frame_width, const int frame_height)
	{
		try{
			FileStorage cvfs;
			cvfs.open(configFile_path, CV_STORAGE_READ);

			//-----------------------------------------------------------------
			loadVisualWords(cvfs);
			//-----------------------------------------------------------------
			loadObjectDB(cvfs);
			//-----------------------------------------------------------------
			getObjRecImgBuf(cvfs, frame_width, frame_height);
			//-----------------------------------------------------------------

			Mat camera_matrix;
			FileStorage fs(cvfs["camera_matrix"], FileStorage::READ);
			fs["camera_matrix"] >> camera_matrix;
			model->init(Size(frame_width, frame_height), camera_matrix);

			model->setMirrorMode(true);	
		}
		catch(std::exception e){
			cout << "Failed to read file " + configFile_path << endl;
			throw e;
		}
	}

///////////////////////////////////////////////////////////////////////////////

	void idleFunc()
	{
		//再描画要求
		glutPostRedisplay();
	}

	void resizeFunc(int w, int h) {
		model->resize(w,h);
	}



	/////////////////////////////////

	boolean test_func(boolean test)
	{
		if (test == true)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	boolean test_start = false;
	//long debug_fcount = -100;
	long debug_fcount = 0;

	void displayFunc(void)
	{
		/////////////////////
		/* 01. open camera */
		/////////////////////
		long start, end;
		Mat frame;

		int start_identify;

		cap >> frame;
		debug_fcount++;

		cout << "debug_fcount: " << debug_fcount << endl;

		//if (debug_fcount == 0)
		//{
		//	start_identify = GetTickCount();
		//}

		////////////////////
		/* 02. 获得图像mat */
		////////////////////
		Mat grayImg;
		cvtColor(frame, grayImg, CV_BGR2GRAY);


		////////////////////
		/* 03. 识别图像    */
		////////////////////
		vector<resultInfo> recog_result;

		if(!track_f)
		{
			try{
				start = GetTickCount();
				cv::resize(grayImg, image_buf, image_buf.size());

				//////////////////////////////
				/* 04. Identify object      */
				//////////////////////////////
				//if (debug_fcount >= 0)
				//{
					recog_result = ctrlOR->queryImage(image_buf);
				//}

				end = GetTickCount();
				cout << "Monitoring: " << " cost: " << end-start <<" ms" << endl;

				if(!recog_result.empty())
				{
					int end_identify = GetTickCount();
					cout << "Time Cost of Object Identification: " << end_identify-start_identify <<" ms" << endl;
					
					debug_fcount = -100;


					Mat pose_mat_scale = recog_result[0].pose_mat.clone();
					pose_mat_scale.row(0) *= query_scale;
					pose_mat_scale.row(1) *= query_scale;


					//--------------------------------------------------------------------------------------------------
					//-------------------------------------- 05. 开始跟踪 ---------------------------------------------
					//--------------------------------------------------------------------------------------------------

					trckOBJ->startTracking(grayImg, scalePoints( recog_result[0].object_position,  (double)query_scale) );
					track_f = true;

					// Jie:　找到了跟踪点，然后初始化model，到了显示model的时候了。
					seq_id = 0;
					wait_seq_id = 0;
				}
			}
			catch(exception e){
			}
		}
		else{
			/*
			* Tracking.
			*
			*/
			start = GetTickCount();
			track_f = trckOBJ->onTracking(grayImg);
			seq_id++;

			if (track_f){
				end = GetTickCount();
				cout << debug_fcount <<" Tracking: " << " cost: " << end - start << " ms" << endl;

				
				std::vector<cv::Point2f> object_postion = trckOBJ->getObjectPosition();
				drawLoLoModel(frame, object_postion);
			}
		}

		////////////////////////////////////////////////////////////////////		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		model->drawScene(frame);
		////////////////////////////////////////////////////////////////////

		glutSwapBuffers();

	}

///////////////////////////////////////////////////////////////////////////////

	int initAR(const int cols, const int rows)
	{
		model = arModel::getInstance();

		// tracking obj 对象
		// Kanade-Lucas-Tomasi(KLT)进行目标跟踪
		trckOBJ = trackingOBJ::create(trackingOBJ::TRACKER_KLT);

		// Load configure file and set parameters.
		setARConfig(cols, rows);	//#
		

		if(ctrlOR == 0) 
		{
			cout << "ctrlOR == 0" << endl;
			return -1;
		}

		return 0;
	}

	int openGUI_openGL(int argc, char *argv[])
	{
		//1. Init capture.
		if( !cap.isOpened() ) {
			cout << "Fail to Open Camera" << endl;
			return -1;
		}

		Mat	frame;
		cap >> frame;

		//2. Init OpenGL
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(frame.cols, frame.rows);
		glutCreateWindow("COMP9517 AR SHOW");

		//3. Init AR
		initAR(frame.cols, frame.rows);

		//4. Init callback
		glutDisplayFunc(displayFunc);
		glutReshapeFunc(resizeFunc);
		glutIdleFunc(idleFunc);

		//5. Run
		glutMainLoop();

		arModel::deleteInstance();

		return 0;
	}

};
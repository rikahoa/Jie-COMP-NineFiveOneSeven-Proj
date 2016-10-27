#include "viewModel.h"
#include "commonCvFunctions.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GLMetaseq.h"
#include "modelobject.h"


using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::overlay;

arModel* arModel::vmInstance = 0;

arModel::arModel(void)
{
	two_power_width = 1024;
	two_power_height = 1024;
	focal_length = 1.0;
	mirror_f = false;
	wait_frames = -1;
	mat_type = -1;
}

arModel::~arModel(void)
{
	releaseModel();
	releaseWaitModel();
}


void arModel::exitFunc()
{
	releaseModel();
	releaseWaitModel();
	glDeleteTextures(1, &texture[0]);
	mat_type = -1;
}

void arModel::setMirrorMode(bool flag)
{
	this->mirror_f = flag;
	if(mat_type==CV_32FC1){
		float td[] = {-1, 0, capture_width-1, 0, 1, 0, 0, 0, 1};
		mirrorMat = Mat(3, 3, mat_type, td).clone();
	}
	else{
		double td[] = {-1, 0, capture_width-1, 0, 1, 0, 0, 0, 1};
		mirrorMat = Mat(3, 3, mat_type, td).clone();
	}
}


void arModel::setFocalLength(float len)
{
	this->focal_length = len;
}


void arModel::releaseModel()
{
	map<int,MODEL_INFO>::iterator itr = model_map.begin();
	while(itr != model_map.end())
	{
		itr->second.model->release();
		delete itr->second.model;
		itr++;
	}
	model_map.clear();
}




void arModel::releaseWaitModel()
{
	if(wait_frames >=0 ){
		wait_model.model->release();
		delete wait_model.model;
		wait_frames = -1;
	}
}


void arModel::initAccHomMat()
{
	if(mat_type == CV_32FC1){
		float f[] = {1,0,0,0,1,0,0,0,1};
		this->accHomMat = Mat(3,3,CV_32FC1,f).clone();
	}
	else if(mat_type == CV_64FC1){
		double d[] = {1,0,0,0,1,0,0,0,1};
		this->accHomMat = Mat(3,3,CV_64FC1,d).clone();
	}
}


bool arModel::setRecogId(int id, Mat& homMat)
{
	map<int, MODEL_INFO>::iterator itr = model_map.find(id);
	
	if(itr==model_map.end())
	{
		return false;
	}
	else{
		curModel = &(itr->second);

		homMat.convertTo(accHomMat, mat_type);
		if(mirror_f)
		{
			Mat markerMirrorMat = mirrorMat.clone();

			if(mat_type==CV_32FC1)
			{
				markerMirrorMat.at<float>(0,2) = curModel->markerSize.width - 1;
			}
			else
			{
				markerMirrorMat.at<double>(0,2) = curModel->markerSize.width - 1;
			}
			accHomMat = mirrorMat * accHomMat * markerMirrorMat;
		}
		return true;
	}
}

bool arModel::init(Size& cap_size, Mat& cameraMat)
{
	return init(cap_size, cameraMat, cameraMat.type());
}

bool arModel::init(Size& cap_size, Mat& cameraMat, int type)
{
//	assert(cameraMat.type() == CV_32FC1 || cameraMat.type() == CV_64FC1);
	assert(type == CV_32FC1 || type == CV_64FC1);

	mat_type = type;

	capture_width = cap_size.width;
	capture_height = cap_size.height;

	window_width = capture_width;
	window_height = capture_height;

	cameraMat.convertTo(cameraMatrix, mat_type);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glGenTextures(1, &texture[0]);
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	aspect_rate = (double)capture_width / (double)capture_height;
	
	resized_frame.create(two_power_width, two_power_height, CV_8UC3);

	glViewport(0, 0, window_width, window_height);

	return true;
}


void arModel::drawScene(cv::Mat& img)
{
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();

	glOrtho(-aspect_rate, aspect_rate, -1.0, 1.0, -1.0, 1.0);
	

	glDisable(GL_DEPTH_TEST);

	updateTexture(img);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	{
		glTexCoord2d(0.0, 0.0);		glVertex2d(-aspect_rate, -1);
		glTexCoord2d(0.0, 1.0);		glVertex2d(-aspect_rate,  1);
		glTexCoord2d(1.0, 1.0);		glVertex2d( aspect_rate,  1);
		glTexCoord2d(1.0, 0.0);		glVertex2d( aspect_rate, -1);
	}
	glEnd();
	glDisable(GL_TEXTURE_2D);

	glPopMatrix();
}


void setLight(void)
{
	
}


void arModel::resize(int w, int h)
{
	int sx, sy;

	if((float)w/h > aspect_rate){
		window_width = aspect_rate * h;
		window_height = h;
		sx = (w - window_width) / 2;
		sy = 0;
	}
	else{
		window_width = w;
		window_height = w / aspect_rate;
		sx = 0;
		sy = (h - window_height) / 2;
	}

	glViewport(sx, sy, window_width, window_height);
}


// Jie
void arModel::updateTexture(Mat& frame)
{
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	Mat img;

	if(mirror_f){
		cv::flip(frame, img, -1);
	}
	else{
		cv::flip(frame, img, 0);
	}

	cv::resize(img, resized_frame, resized_frame.size());

	/////////////////////////////////////////////////////

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
		resized_frame.cols,resized_frame.rows,
		0, GL_BGR_EXT, GL_UNSIGNED_BYTE, resized_frame.data);
}


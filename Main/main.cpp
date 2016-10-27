#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include "commonCvFunctions.h"
#include "guiAR.h"
#include <io.h>
#include <vector>
#include <random>

using namespace std;
using namespace cv;
using namespace cv9417;
using namespace cv9417::or;

#define CREATE_VW	"==> 1. Create bag of visual words."
#define REGISTF		"==> 2. Analyze Recognized Image."

/*
 * z5004703
 * COMP9517
 * Jie: Set the number of negative samples which will be chosen randomly
 * in training set.   {.\img\u(i).jpg}
 */
int sample_size = 20;

/* load training images */
void getFiles( string path, string exd, vector<string>& files )
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}
	
	if((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
			}
			else
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
				{
					string fileName = fileinfo.name;
					if (fileName[0] == 'u')
					{
						files.push_back(pathName.assign(path).append("\\").append(fileName));
					}
					else
					{
						continue;
					}
				}
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

int main(int argc, char * argv[])
{
	controlOR	controlObjectRecognition;
	setControlOR(controlObjectRecognition);

	////////////////////////////////////////////////////////////////////
	// Step1. Construction of Bag of visual words. 
	////////////////////////////////////////////////////////////////////
	bool exitflag1 = false;
	string opt;
	clock_t start_time, end_time;

	string cmd_arr[10];
	cmd_arr[0] = CREATE_VW;
	cmd_arr[1] = "save_vw_bin";
	cmd_arr[2] = ".\\config\\visualWord.bin";
	cmd_arr[3] = ".\\config\\vw_index.txt";
	cmd_arr[4] = REGISTF;
	cmd_arr[5] = "save_objectDB";
	cmd_arr[6] = ".\\config\\db.txt";
	cmd_arr[7] = "exit";



	int cmd_index = 0;

	while(!exitflag1)
	{
		opt = cmd_arr[cmd_index++];

		// Jie01: Create Visual Word
		if(opt==CREATE_VW)
		{
			cout << opt << endl;

			string opt2;
			int cls_num = 0;

			cout << "Load Image files: " << endl;
			//cin >> opt2;
			//Debug.


			//func();

			///////////////////////////////////////
			char * filePath = ".\\img";
			vector<string> files;

			// 获取该路径下的所有jpg文件
			getFiles(filePath, "jpg", files);

			int size = files.size();
			///////////////////////////////////////


			bool exitflag2 = false;
			int count = 0;
			string buf;

			for (int i = -1; i < sample_size;i++)
			{
				if (i == -1)
				{
					buf = ".\\img\\A1.jpg";
				}
				else
				{
					std::random_device rd;
					std::mt19937 gen(rd());
					std::uniform_int_distribution<> dis(0, size);
					int fileIndex = dis(gen);
					buf = files[fileIndex].c_str();
				}

				cout << buf;

				Mat img_training = imread(buf, 0);
				//imshow(studyimg);
				//start_time = clock();

				/* 1. 提取一幅图像的"描述子" */
				controlObjectRecognition.addFeaturesForVW(img_training);
				//end_time = clock();
				//cout << ((double)end_time - start_time)/CLOCKS_PER_SEC << endl;
				cout << endl;
			}

			//start_time = clock();

			/* 2. 添加所有图片的描述符（到FlannBasedMatcher）之后，然后就开始训练。
			*    形成了 Bag of visual words and clusters。
			*/
			controlObjectRecognition.createVisualWords(cls_num);	// <--

			//end_time = clock();

			//cout << ((double)end_time - start_time)/CLOCKS_PER_SEC << endl;
			cout << endl;
		}
		else if(opt=="save_vw_bin")
		{
			string filename, idxname;
			filename = cmd_arr[cmd_index++];
			idxname = cmd_arr[cmd_index++];
			controlObjectRecognition.saveVisualWordsBinary(filename,idxname);
		}
		else if(opt== REGISTF)
		{
			cout << opt << endl;

			string regist_list;
			cout << "Recognized Image: ";
			string imgfile = ".\\img\\R1.jpg";
			cout << imgfile;
			Mat studyimg = imread(imgfile, 0);

			// 注册图片
			controlObjectRecognition.registImage(studyimg, 1);
			cout << endl;

		}
		else if(opt=="save_objectDB")
		{
			string filename;
			filename = cmd_arr[cmd_index++];
			controlObjectRecognition.saveObjectDB(filename);
			cout << endl;
		}
		else if(opt=="exit"){
			break;
		}
		else{
			cout << "Error: Wrong Command\n" << endl;
		}
	} // while end.

	////////////////////////////////////////////////////////////////////
	// Step2. Start AR.
	////////////////////////////////////////////////////////////////////
	string configPath(argv[1]);
	setConfigFile(configPath);
	openGUI_openGL(argc, argv);

	return 0;
}

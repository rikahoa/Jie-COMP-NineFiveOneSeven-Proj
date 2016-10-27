#ifndef __MODELOBJECT__
#define __MODELOBJECT__

#include <string>

namespace cv9417{
	namespace overlay{

		class modelObject{
		public:
			virtual ~modelObject(void){}

			virtual void init() = 0;
			virtual void loadModelFile(std::string filename) = 0;
			virtual void drawModel(int& frame_id) = 0;
			virtual void release() = 0;

		public:
			int status;
			static const int UNINIT = 0x0000;
			static const int INIT = 0x0001;
			static const int LOADED = 0x0002;
		};

	};
};
#endif

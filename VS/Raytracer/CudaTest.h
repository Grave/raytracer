#ifndef __CUDA_TEST__
#define __CUDA_TEST__

#include <vector> 

using std::vector; 

class CudaTest
{

	public:
		CudaTest(void)
		{}
		~CudaTest(void)
		{}

		void InitializeData(vector<float>& data);
		void RunCubeKernel(vector<float>& data, vector<float>& result);

		static const int MaxSize; 
	private:
};

#endif // __CUDA_TEST__
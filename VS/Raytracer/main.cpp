#include <iostream> 
#include <vector> 
#include "CudaTest.h"

using std::cout; 
using std::endl; 
using std::vector; 

int main(void) 
{ 
	cout << "In main." << endl; 

	CudaTest test;

	// Create sample data 
	vector<float> data(CudaTest::MaxSize); 
	test.InitializeData(data); 

	// Compute cube on the device 
	vector<float> cube(CudaTest::MaxSize); 
	test.RunCubeKernel(data, cube); 

	// Print out results 
	cout << "Cube kernel results." << endl << endl; 

	for (int i = 0; i < CudaTest::MaxSize; ++i) 
	{ 
		cout << cube[i] << endl; 
	} 

	return 0; 
}
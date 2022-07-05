#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"

using namespace std;


int main(int argc, char* argv[])
{
	try
	{
		trainReaderInputs("MNIST/train-images.idx3-ubyte");
	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
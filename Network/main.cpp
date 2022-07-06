#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"
#include "Network.h"

using namespace std;


int main(int argc, char* argv[])
{
	try
	{
		//Matrix inputs = trainReaderInputs("MNIST/train-images.idx3-ubyte");
		//Matrix labels = trainReaderLabels("MNIST/train-labels.idx1-ubyte");
		//Network network({ { 784, 15, 10 } });
		//network.loadInputs(inputs);
		//network.loadDesiredOutput(labels);


	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
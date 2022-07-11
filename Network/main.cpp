#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"
#include "Network.h"

using namespace std;

#define TRAIN 1


int main(int argc, char* argv[])
{
	try
	{
#if TRAIN
		std::unique_ptr<Matrix[]> p;
		Matrix inputs = trainReaderInputs("MNIST/train-images.idx3-ubyte");
		//Matrix labels = trainReaderLabels("MNIST/train-labels.idx1-ubyte");
		Network network({ { 784, 15, 10 } });
		//network.loadInputs(inputs);
		//convertLabelToMatrixArray(labels, &p);
		//network.loadDesiredOutput(&p, labels.get_size().rows);
		//network.SGD(0.05, 1000, 60);

		inputs = Matrix::t(inputs.get_row(1));
		network.loadInputs(inputs);
//#else
		network.readNetworkWeightsAndBiases("network_data.txt");
		cout << network.evaluateNetwork() << endl;
#endif

	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
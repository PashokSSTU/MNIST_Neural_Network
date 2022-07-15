#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"
#include "Network.h"

using namespace std;

#define TRAIN 1
#define ADDITIONAL_TRAINING 0


int main(int argc, char* argv[])
{
	try
	{
		Matrix layers = { {2, 1} };
		Network network(layers);

#if TRAIN

#if ADDITIONAL_TRAINING
		network.readNetworkWeightsAndBiases("network_data.txt");
#endif
		std::unique_ptr<Matrix[]> arr_labels;
		std::unique_ptr<Matrix[]> arr_test_labels;

		Matrix inputs = trainReaderInputs("MNIST/t10k-images.idx3-ubyte");
		Matrix labels = trainReaderLabels("MNIST/t10k-labels.idx1-ubyte");
		Matrix test_inputs = testReaderInputs("MNIST/train-images.idx3-ubyte");
		Matrix test_labels = testReaderLabels("MNIST/train-labels.idx1-ubyte");

		convertLabelToMatrixArray(labels, &arr_labels);
		convertLabelToMatrixArray(test_labels, &arr_test_labels);

		network.loadTrainingInputs(inputs);
#else
		
		network.readNetworkWeightsAndBiases("network_data.txt");

		Matrix inputs =
		{
			{0},
			{0}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetwork() << endl << endl;

		inputs =
		{
			{0},
			{1}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetwork() << endl << endl;

		inputs =
		{
			{1},
			{0}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetwork() << endl << endl;

		inputs =
		{
			{1},
			{1}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetwork() << endl << endl;

#endif

	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
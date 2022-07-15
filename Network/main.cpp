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
		Matrix layers = { {784, 15, 10} };
		Network network(layers);

#if TRAIN

#if ADDITIONAL_TRAINING
		network.readNetworkWeightsAndBiases("network_data.txt");
#endif
		std::unique_ptr<Matrix[]> arr_labels;
		std::unique_ptr<Matrix[]> arr_test_labels;

		Matrix inputs = trainReaderInputs("MNIST/train-images.idx3-ubyte");
		Matrix labels = trainReaderLabels("MNIST/train-labels.idx1-ubyte");
		Matrix test_inputs = testReaderInputs("MNIST/t10k-images.idx3-ubyte");
		Matrix test_labels = testReaderLabels("MNIST/t10k-labels.idx1-ubyte");

		convertLabelToMatrixArray(labels, &arr_labels);
		convertLabelToMatrixArray(test_labels, &arr_test_labels);

		network.loadTrainingInputs(inputs);
		network.loadTestInputs(test_inputs);
		network.loadDesiredTrainingOutputs(&arr_labels, labels.get_size().rows);
		network.loadDesiredTestOutputs(&arr_test_labels, test_labels.get_size().rows);

		network.SGD(3.0, 10, 30, true);
#else
		
		network.readNetworkWeightsAndBiases("network_data.txt");

		Matrix inputs =
		{
			{0},
			{0}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetworkOutput() << endl << endl;

		inputs =
		{
			{0},
			{1}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetworkOutput() << endl << endl;

		inputs =
		{
			{1},
			{0}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetworkOutput() << endl << endl;

		inputs =
		{
			{1},
			{1}
		};

		network.loadInputs(inputs);
		cout << "Input: " << endl << inputs << endl << "evaluate: " << network.evaluateNetworkOutput() << endl << endl;

#endif

	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"
#include "Network.h"

using namespace std;

#define TRAIN 0


int main(int argc, char* argv[])
{
	try
	{
		
		Network network({ { 2, 5, 1 } });

#if TRAIN
		std::unique_ptr<Matrix[]> p;

		Matrix inputs = {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};

		p = make_unique<Matrix[]>(inputs.get_size().rows);

		p[0] = { {0} };
		p[1] = { {0} };
		p[2] = { {1} };
		p[3] = { {1} };


		network.loadInputs(inputs);
		network.loadDesiredOutput(&p, inputs.get_size().rows);
		network.SGD(0.05, 4, 100000);

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
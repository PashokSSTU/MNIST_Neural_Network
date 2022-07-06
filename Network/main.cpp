#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"
#include "Network.h"

using namespace std;


int main(int argc, char* argv[])
{
	try
	{
		Network network({ {2, 4, 1} });

		Matrix inputs = {
			{1},
			{2}
		};

		Matrix W1 =
		{
			{2, 0},
			{10, -5},
			{1, 1},
			{4, 6}
		};

		Matrix b1 =
		{
			{0},
			{1},
			{6},
			{-5}
		};

		Matrix W2 =
		{
			{1, 2, 0, 1}
		};

		Matrix b2 =
		{
			{1}
		};

		cout << network.sigmoid(W2 * (network.sigmoid(W1 * inputs + b1)) + b2) << endl;
		network.loadInputs(inputs);
		cout << network.evaluateNetwork() << endl;

		


	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
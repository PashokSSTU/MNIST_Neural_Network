#include "Network.h"

Network::Network(const Matrix& _layers) : layers(_layers.get_size().columns)
{
	//It is necessary to randomize the filling of weights and biases!
	//-------------------------------------------------------------//
	weights = std::make_unique<Matrix[]>(_layers.get_size().columns - 1);
	biases = std::make_unique<Matrix[]>(_layers.get_size().columns - 1);
	//-------------------------------------------------------------//

	for (int i = 0; i < _layers.get_size().columns - 1; i++)
	{
		weights[i] = Matrix(_layers.get_elem(1, i + 2), _layers.get_elem(1, i + 1));
	}

	for (int i = 0; i < _layers.get_size().columns - 1; i++)
	{
		biases[i] = Matrix(_layers.get_elem(1, i + 2), 1);
	}
}

Network::~Network() {}

double Network::sigmoid(double x)
{
	return (1 / (1 + std::exp(-x)));
}

double Network::sigmoid_derivative(double x)
{
	return (sigmoid(x) * (1 - sigmoid(x)));
}

Matrix Network::sigmoid(const Matrix& matrix)
{
	Matrix result(matrix.get_size());
	for (int i = 1; i <= matrix.get_size().rows; i++)
	{
		for (int j = 1; j <= matrix.get_size().columns; j++)
		{
			result.set_elem(sigmoid(matrix.get_elem(i, j)), i, j);
		}
	}

	return result;
}

Matrix Network::sigmoid_derivative(const Matrix& matrix)
{
	Matrix result(matrix.get_size());
	for (int i = 1; i <= matrix.get_size().rows; i++)
	{
		for (int j = 1; j <= matrix.get_size().columns; j++)
		{
			result.set_elem(sigmoid_derivative(matrix.get_elem(i, j)), i, j);
		}
	}

	return result;
}

Matrix Network::getLayer_weights(int l)
{
	return weights[l - 1];
}

Matrix Network::getLayer_biases(int l)
{
	return biases[l - 1];
}

Matrix Network::evaluateLayer(int l)
{
	if (inputs == nullptr)
	{
		throw ERROR_INPUTS_DIDNT_LOAD;
	}
	Matrix activation = (*inputs);

	for (int _l = 2; _l <= l; _l++)
	{
		activation = sigmoid((weights[_l - 2] * activation) + biases[_l - 2]);
	}

	return activation;
}

Matrix Network::evaluateNetwork()
{
	return evaluateLayer(layers);
}

void Network::loadInputs(const Matrix& input)
{
	inputs = std::make_unique<Matrix>(input.get_size());
	for (int i = 1; i <= input.get_size().rows; i++)
	{
		for (int j = 1; j <= input.get_size().columns; j++)
		{
			inputs->set_elem(input.get_elem(i, j), i, j);
		}
	}
}

void Network::loadDesiredOutput(const Matrix& output)
{
	desired_outputs = std::make_unique<Matrix>(output.get_size());
	for (int i = 1; i <= output.get_size().rows; i++)
	{
		for (int j = 1; j <= output.get_size().columns; j++)
		{
			desired_outputs->set_elem(output.get_elem(i, j), i, j);
		}
	}
}
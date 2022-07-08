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

	if (l == 1)
	{
		return (*inputs);
	}

	Matrix activation = (*inputs);

	for (int _l = 2; _l <= l; _l++)
	{
		activation = sigmoid((weights[_l - 2] * activation) + biases[_l - 2]);
	}

	return activation;
}

Matrix Network::evaluateInputsOfLayer(int l)
{
	if (inputs == nullptr)
	{
		throw ERROR_INPUTS_DIDNT_LOAD;
	}
	Matrix z = (*inputs);

	for (int _l = 2; _l <= l; _l++)
	{
		if(_l != l)
			z = sigmoid((weights[_l - 2] * z) + biases[_l - 2]);
		else
			z = (weights[_l - 2] * z) + biases[_l - 2];
	}

	return z;
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

void Network::loadDesiredOutput(const std::unique_ptr<Matrix[]>* p, int amounth)
{
	desired_outputs = std::make_unique<Matrix[]>(amounth);
	for (int i = 0; i < amounth; i++)
	{
		desired_outputs[i] = (*p)[i];
	}
}

void Network::training_shuffle()
{
	srand(time(nullptr));

	for (int i = 60000 - 1; i >= 1; i--)
	{
		int j = rand() % (i + 1);

		Matrix tmp_lbl = desired_outputs[j];
		desired_outputs[j] = desired_outputs[i];
		desired_outputs[i] = tmp_lbl;

		Matrix tmp_inp = inputs->get_row(j + 1);
		inputs->set_row(inputs->get_row(i + 1), j + 1);
		inputs->set_row(tmp_inp, i + 1);
	}
}


void Network::get_mini_batch(int mini_batch_size)
{
	mini_batches.clear();

	for (int i = 0; i < mini_batch_size; i++)
	{
		mini_batch_elem.inp = Matrix::t(inputs->get_row(i + 1));
		mini_batch_elem.out = desired_outputs[i];

		mini_batches.push_back(mini_batch_elem);
	}
}

void Network::update_mini_batch(double eta)
{
	int train_number = 0;
	dnw = std::make_unique<Matrix[]>(layers - 1);
	dnb = std::make_unique<Matrix[]>(layers - 1);
	
	for (int l = layers - 1; l >= 2; l--)
	{
		dnw[l - 2] = Matrix::Zeros(weights[l].get_size().rows, weights[l - 2].get_size().columns);
		dnb[l - 2] = Matrix::Zeros(biases[l].get_size().rows, biases[l - 2].get_size().columns);
	}

	// Calculating delta's
	for (auto it = mini_batches.begin(); it != mini_batches.end(); it++)
	{
		train_number++;
		backpropogation(train_number);
		for (int l = layers - 1; l >= 2; l--)
		{
			dnw[l - 2] += nabla_w[l - 2];
			dnb[l - 2] += nabla_b[l - 2];
		}
	}

	// Update w and b
	for (int l = layers - 1; l >= 2; l--)
	{
		weights[l - 2] = (eta / mini_batches.size()) * (dnw[l - 2]);
		biases[l - 2] = (eta / mini_batches.size()) * (dnb[l - 2]);
	}
}

Matrix Network::cost_derivative(const Matrix& desired, const Matrix& outputs)
{
	return (outputs - desired);
}

void Network::SGD(double eta, int mini_batch_size, int epohs)
{
	//for (int ep = 0; ep < epohs; ep++)
	//{
	//	for (int inp = 1; inp <= inputs->get_size().rows; inp++)
	//	{
	//		backpropogation(inp); // calculating of nablas for weights and biases
	//	}
	//}
}

void Network::backpropogation(int train_number)
{
	Matrix outputs = evaluateNetwork();
	Matrix z = evaluateInputsOfLayer(layers);
	Matrix delta = Matrix::Hadamard_product(cost_derivative(desired_outputs[train_number - 1], outputs), sigmoid_derivative(z));

	nabla_w = std::make_unique<Matrix[]>(layers - 1);
	nabla_b = std::make_unique<Matrix[]>(layers - 1);

	nabla_b[layers - 2] = delta;
	nabla_w[layers - 2].resize(weights[layers - 2].get_size());
	Matrix a = evaluateLayer(layers - 1);
	nabla_w[layers - 2] = a * Matrix::t(delta);

	for (int l = layers - 1; l >= 2; l--)
	{
		z = evaluateInputsOfLayer(l);
		delta = Matrix::Hadamard_product(weights[l - 1] * delta, sigmoid_derivative(z));
		nabla_b[l - 2] = delta;
		nabla_w[l - 2] = evaluateLayer(l - 1) * Matrix::t(delta);
	}
}
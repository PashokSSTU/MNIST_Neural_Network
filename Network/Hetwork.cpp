#include "Network.h"

Network::Network(const Matrix& _layers) : layers(_layers.get_size().columns)
{
	weights = std::make_unique<Matrix[]>(_layers.get_size().columns - 1);
	biases = std::make_unique<Matrix[]>(_layers.get_size().columns - 1);

	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> d{ 0.0,1.0 };

	for (int i = 0; i < _layers.get_size().columns - 1; i++)
	{
		weights[i] = Matrix(_layers.get_elem(1, i + 2), _layers.get_elem(1, i + 1));
	}

	for (int i = 0; i < _layers.get_size().columns - 1; i++)
	{
		biases[i] = Matrix(_layers.get_elem(1, i + 2), 1);
	}

	// random initialization of weights and biases
	for (int l = 0; l < layers - 1; l++)
	{
		for (int i = 1; i <= weights[l].get_size().rows; i++)
		{
			for (int j = 1; j <= weights[l].get_size().columns; j++)
			{
				weights[l].set_elem(d(gen), i, j);
			}
		}

		for (int i = 1; i <= biases[l].get_size().rows; i++)
		{
			for (int j = 1; j <= biases[l].get_size().columns; j++)
			{
				biases[l].set_elem(d(gen), i, j);
			}
		}
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
	return weights[l - 2];
}

Matrix Network::getLayer_biases(int l)
{
	return biases[l - 2];
}

Matrix Network::evaluateLayer(int l, int train_number)
{
	if (inputs == nullptr)
	{
		throw ERROR_INPUTS_DIDNT_LOAD;
	}

	if (l == 1)
	{
		if (train_number)
			return Matrix::t(inputs->get_row(train_number));
		else
			return (*inputs);
	}

	Matrix activation;

	if (train_number)
		activation = Matrix::t(inputs->get_row(train_number));
	else
		activation = (*inputs);

	for (int _l = 2; _l <= l; _l++)
	{
		activation = sigmoid((weights[_l - 2] * activation) + biases[_l - 2]);
	}

	return activation;
}

Matrix Network::evaluateInputsOfLayer(int l, int train_number)
{
	if (inputs == nullptr)
	{
		throw ERROR_INPUTS_DIDNT_LOAD;
	}

	Matrix z;

	if (train_number)
		z = Matrix::t(inputs->get_row(train_number));
	else
		z = (*inputs);

	for (int _l = 2; _l <= l; _l++)
	{
		if (_l == l)
		{
			z = (weights[_l - 2] * z) + biases[_l - 2];
		}
		else
		{
			z = sigmoid((weights[_l - 2] * z) + biases[_l - 2]);
		}
	}

	return z;
}

Matrix Network::evaluateNetwork(int train_number)
{
	return evaluateLayer(layers, train_number);
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

	for (int i = inputs->get_size().rows - 1; i >= 1; i--)
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


void Network::get_mini_batch(int& start_batch_number, int mini_batch_size)
{
	mini_batches.clear();

	for (int i = start_batch_number; i < start_batch_number + mini_batch_size; i++)
	{
		if (i >= inputs->get_size().rows)
		{
			i -= start_batch_number;
			start_batch_number = 0;
		}

		mini_batch_elem.inp = Matrix::t(inputs->get_row(i + 1));
		mini_batch_elem.out = desired_outputs[i];

		mini_batches.push_back(mini_batch_elem);
	}

	start_batch_number += mini_batch_size;
}

void Network::update_mini_batch(double eta)
{
	int train_number = 0;
	dnw = std::make_unique<Matrix[]>(layers - 1);
	dnb = std::make_unique<Matrix[]>(layers - 1);
	
	for (int l = layers; l >= 2; l--)
	{
		dnw[l - 2].resize(weights[l - 2].get_size());
		dnb[l - 2].resize(biases[l - 2].get_size());

	}

	// Calculating delta's
	for (auto it = mini_batches.begin(); it != mini_batches.end(); it++)
	{
		train_number++;
		backpropogation(train_number);
		for (int l = layers; l >= 2; l--)
		{
			dnw[l - 2] += nabla_w[l - 2];
			dnb[l - 2] += nabla_b[l - 2];
		}
	}

	// Update w and b
	for (int l = layers; l >= 2; l--)
	{
		weights[l - 2] -= (eta / mini_batches.size()) * (dnw[l - 2]);
		biases[l - 2] -= (eta / mini_batches.size()) * (dnb[l - 2]);
	}
}

void Network::saveNetworkWeightsAndBiases(const char* filepath)
{
	std::ofstream f(filepath, std::ios::out | std::ios::binary);
	if (f.is_open())
	{
		for (int l = 0; l < layers - 1; l++)
		{
			double tmp;
			tmp = weights[l].get_size().rows;
			f.write(reinterpret_cast<char*>(&tmp), sizeof(double));
			tmp = weights[l].get_size().columns;
			f.write(reinterpret_cast<char*>(&tmp), sizeof(double));

			for (int i = 1; i <= weights[l].get_size().rows; i++)
			{
				for (int j = 1; j <= weights[l].get_size().columns; j++)
				{
					tmp = weights[l].get_elem(i, j);
					f.write(reinterpret_cast<char*>(&tmp), sizeof(double));
				}
			}
		}

		for (int l = 0; l < layers - 1; l++)
		{
			double tmp;
			tmp = biases[l].get_size().rows;
			f.write(reinterpret_cast<char*>(&tmp), sizeof(double));
			tmp = biases[l].get_size().columns;
			f.write(reinterpret_cast<char*>(&tmp), sizeof(double));

			for (int i = 1; i <= biases[l].get_size().rows; i++)
			{
				for (int j = 1; j <= biases[l].get_size().columns; j++)
				{
					tmp = biases[l].get_elem(i, j);
					f.write(reinterpret_cast<char*>(&tmp), sizeof(double));
				}
			}

		}

		f.close();
	}
	else
	{
		throw "Error of creating network's data save file!";
	}
}

void Network::readNetworkWeightsAndBiases(const char* filepath)
{
	std::ifstream f(filepath, std::ios::in | std::ios::binary);
	if (f.is_open())
	{
		for (int l = 0; l < layers - 1; l++)
		{
			double tmp;
			Matrix m_tmp;
			f.read((char*)&tmp, sizeof(double));
			int row = tmp;
			f.read((char*)&tmp, sizeof(double));
			int column = tmp;
			m_tmp.resize(row, column);
			for (int i = 1; i <= weights[l].get_size().rows; i++)
			{
				for (int j = 1; j <= weights[l].get_size().columns; j++)
				{
					f.read((char*)&tmp, sizeof(double));
					m_tmp.set_elem(tmp, i, j);
				}
			}
			weights[l] = m_tmp;
		}

		for (int l = 0; l < layers - 1; l++)
		{
			double tmp;
			Matrix m_tmp;
			f.read((char*)&tmp, sizeof(double));
			int row = tmp;
			f.read((char*)&tmp, sizeof(double));
			int column = tmp;
			m_tmp.resize(row, column);
			for (int i = 1; i <= biases[l].get_size().rows; i++)
			{
				for (int j = 1; j <= biases[l].get_size().columns; j++)
				{
					f.read((char*)&tmp, sizeof(double));
					m_tmp.set_elem(tmp, i, j);
				}
			}
			biases[l] = m_tmp;
		}

		f.close();
	}
	else
	{
		throw "Error: \"Network's data file didn't fing!\"";
	}
}

Matrix Network::cost_derivative(const Matrix& desired, const Matrix& outputs)
{
	return (outputs - desired);
}

void Network::SGD(double eta, int mini_batch_size, int epohs)
{
	training_shuffle();
	int start_batch = 0;
	for (int ep = 0; ep < epohs; ep++)
	{
		std::cout << "Epoch " << ep + 1 << " started" << std::endl;
		get_mini_batch(start_batch, mini_batch_size);
 		update_mini_batch(eta);
		std::cout << "Epoch " << ep + 1 << " ended" << std::endl;
	}

	saveNetworkWeightsAndBiases("network_data.txt");
}

void Network::backpropogation(int train_number)
{
	Matrix outputs = evaluateNetwork(train_number);
	Matrix z = evaluateInputsOfLayer(layers, train_number);
	Matrix delta = Matrix::Hadamard_product(cost_derivative(desired_outputs[train_number - 1], outputs), sigmoid_derivative(z));

	nabla_w = std::make_unique<Matrix[]>(layers - 1);
	nabla_b = std::make_unique<Matrix[]>(layers - 1);

	nabla_b[layers - 2] = delta;
	nabla_w[layers - 2].resize(weights[layers - 2].get_size());
	Matrix a = evaluateLayer(layers - 1, train_number);
	nabla_w[layers - 2] = delta * Matrix::t(a);

	for (int l = layers - 1; l >= 2; l--)
	{
		z = evaluateInputsOfLayer(l, train_number);
		delta = Matrix::Hadamard_product(Matrix::t(weights[l - 1]) * delta, sigmoid_derivative(z));
		nabla_b[l - 2] = delta;
		nabla_w[l - 2] = delta * Matrix::t(evaluateLayer(l - 1, train_number));
	}
}
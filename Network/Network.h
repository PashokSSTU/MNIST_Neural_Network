#pragma once
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <list>
#include <cmath>
#include <random>
#include <cfloat>
#include "Matrix.h"

#define ERROR_INPUTS_DIDNT_LOAD "Error! Inputs for network didn't load!"

class Network
{
public:
	Network(const Matrix& _layers);
	~Network();

	double sigmoid(double x);
	double sigmoid_derivative(double x);
	Matrix sigmoid(const Matrix& matrix);
	Matrix sigmoid_derivative(const Matrix& matrix);

	Matrix getLayer_weights(int l);
	Matrix getLayer_biases(int l);

	Matrix evaluateLayer(int l, int train_number = 0, bool test = false);
	Matrix evaluateInputsOfLayer(int l, int train_number = 0);
	Matrix evaluateNetworkOutput(int train_number = 0, bool test = false);
	Matrix evaluateNetwork(int train_number = 0, bool test = false);

	void loadTrainingInputs(const Matrix& input);
	void loadTestInputs(const Matrix& input);
	void loadDesiredTrainingOutputs(const std::unique_ptr<Matrix[]>* p, int amounth);
	void loadDesiredTestOutputs(const std::unique_ptr<Matrix[]>* p, int amounth);

	//Train
	Matrix cost_derivative(const Matrix& desired, const Matrix& outputs);
	void SGD(double eta, int mini_batch_size, int epohs, bool test = false);//gradient desgent
	void backpropogation(int train_number);
	void training_shuffle();
	void get_mini_batch(int& start_batch_number, int mini_batch_size);
	void update_mini_batch(double eta);

	void saveNetworkWeightsAndBiases(const char* filepath);
	void readNetworkWeightsAndBiases(const char* filepath);

private:
	int layers = 0;

	std::unique_ptr<Matrix> inputs = nullptr;
	std::unique_ptr<Matrix> test_inputs = nullptr;
	std::unique_ptr<Matrix[]> weights = nullptr;
	std::unique_ptr<Matrix[]> biases = nullptr;
	std::unique_ptr<Matrix[]> desired_outputs = nullptr;
	std::unique_ptr<Matrix[]> desired_test_outputs = nullptr;

	// mini_batches variables
	struct _mini_batch_elem
	{
		Matrix inp;
		Matrix out;
		int train_number;
	} mini_batch_elem;

	std::list<_mini_batch_elem> mini_batches;

	//Nabla of W and B layers
	std::unique_ptr<Matrix[]> nabla_w = nullptr;
	std::unique_ptr<Matrix[]> nabla_b = nullptr;

	std::unique_ptr<Matrix[]> dnw = nullptr;
	std::unique_ptr<Matrix[]> dnb = nullptr;
};
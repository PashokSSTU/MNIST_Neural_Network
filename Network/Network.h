#pragma once
#include <iostream>
#include <memory>
#include <cmath>
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

	Matrix evaluateLayer(int l);
	Matrix evaluateNetwork();

	void loadInputs(const Matrix& input);

	//Train
	void backpropogation();
	void loadDesiredOutput(const Matrix& output);

private:
	int layers = 0;
	std::unique_ptr<Matrix> inputs = nullptr;
	std::unique_ptr<Matrix[]> weights = nullptr;
	std::unique_ptr<Matrix[]> biases = nullptr;
	std::unique_ptr<Matrix> desired_outputs = nullptr;
};
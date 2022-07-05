#pragma once
#include "Matrix.h"
#include <fstream>

// Exceptions
#define ERROR_OF_OPENING_TRAIN_INPUTS	"Error of opening train inputs"
#define ERROR_OF_OPENING_TRAIN_LABELS	"Error of opening train labels"
#define ERROR_OF_OPENING_TEST_INPUTS	"Error of opening test inputs"
#define ERROR_OF_OPENING_TEST_INPUTS	"Error of opening test labels"

#define TRAIN_INPUTS_INCORRECT_ID		"Error of parsing training images. Number must to be 2051!"
#define TRAIN_LABELS_INCORRECT_ID		"Error of parsing training labels. Number must to be 2049!"
#define TEST_INPUTS_INCORRECT_ID		"Error of parsing test images. Number must to be 2051!"
#define TEST_LABELS_INCORRECT_ID		"Error of parsing test labels. Number must to be 2049!"

#define IT_ISNT_TRAINING_SET			"Error! It isn't training set!"
#define IT_ISNT_TEST_SET				"Error! It isn't test set!"

int ReverseInt(int i);
uint8_t ReverseChar(uint8_t b);

Matrix trainReaderInputs(const char* filepath);
Matrix trainReaderLabels(const char* filepath);
Matrix testReaderInputs(const char* filepath);
Matrix testReaderLabels(const char* filepath);
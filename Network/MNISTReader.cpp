#include "MNISTReader.h"

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

uint8_t ReverseChar(uint8_t b)
{
	b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
	b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
	b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
	return b;
}

Matrix trainReaderInputs(const char* filepath)
{
	Matrix inputs = Matrix::Zeros(60000, 784);

	std::ifstream f(filepath, std::ios::in | std::ios::binary);
	if (f.is_open())
	{
		int data = 0;

		f.read((char*)&data, sizeof(int));
		data = ReverseInt(data);
		if (data != 2051)
		{
			throw TRAIN_INPUTS_INCORRECT_ID;
		}
		
		f.read((char*)&data, sizeof(int));
		data = ReverseInt(data);
		if (data != 60000)
		{
			throw IT_ISNT_TRAINING_SET;
		}

		for (int i = 1; i <= 60000; i++)
		{
			for (int j = 1; j <= 784; j++)
			{
				uint8_t elem = 0;

				f.read((char*)&elem, sizeof(char));
				elem = ReverseChar(elem);
				inputs.set_elem(elem, i, j);
			}
		}

		f.close();
	}
	else
	{
		throw ERROR_OF_OPENING_TRAIN_INPUTS;
	}

	return inputs;
}
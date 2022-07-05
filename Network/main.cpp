#include <iostream>
#include "Matrix.h"
#include "MNISTReader.h"

using namespace std;


int main(int argc, char* argv[])
{
	try
	{
		Matrix dataImages = trainReaderInputs("MNIST/train-images.idx3-ubyte");

		Matrix inputs = dataImages.get_row(1);
		inputs = Matrix::t(inputs);

		cout << inputs << endl;

		ofstream f_out("MNIST/test_num.txt", ios::out);

		if (f_out.is_open())
		{
			for (int i = 1; i <= inputs.get_size().rows; i++)
			{
				if (inputs.get_elem(i, 1) == 0)
				{
					f_out.write("_", sizeof(char));
					if (i % 28 == 0)
					{
						f_out.write("\r\n", sizeof(char));
					}
				}
				else
				{
					f_out.write("#", sizeof(char));
					if (i % 28 == 0)
					{
						f_out.write("\r\n", sizeof(char));
					}
				}
			}

			f_out.close();
		}
		else
		{
			exit(1);
		}

	}
	catch (const char* e)
	{
		cout << e << endl;
	}

	return 0;
}
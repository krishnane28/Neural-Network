#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;

int main()
{
	// myModel defines my neural network model with layers and total neurons in each layer
	// First element is the number of neurons in the input layer
	// Last element is the number of neurons in the output layer
	// Rest will be the number of neurons in the hidden layers
	vector<unsigned int> myModel;
	// Input layer with two neurons
	myModel.push_back(3);
	// Hidden layer with two neurons
	myModel.push_back(3);
	myModel.push_back(3);
	// Output layer with one neuron
	myModel.push_back(1);

	NeuralNetwork myNetwork = NeuralNetwork(myModel);

	for (unsigned int epoch = 0; epoch < 300; epoch++)
	{
		vector<double> inputs{ 1, 2, 4 };

		vector<double> target{ 7 };

		myNetwork.feedForward(inputs, target);

		myNetwork.backPropagation(inputs, target);
	}

	cout << "Test: " << endl;

	vector<double> inputs{ 1, 2, 4 };

	vector<double> target{ 7 };

	myNetwork.feedForward(inputs, target);

	return 0;
}
#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include "Neuron.h"

class NeuralNetwork
{
public:
	// Constructor to create a neural network with myModel which contains the total number of 
	// layers and total neurons in each layer in the network
	NeuralNetwork(const std::vector<unsigned int> &myModel);
	// This does the feed forward process in the neural network and calculates the output
	void feedForward(const std::vector<double> &inputValues,
		const std::vector<double> &targetValues);
	// This does the back propagation for the network i.e. adjust the weights 
	void backPropagation(const std::vector<double> &inputValues, const std::vector<double> &targetValues);
private:
	// This is a two dimensional vector which represents each layer and total neurons in it
	std::vector<Layer> layers;
	// This is the error produced by the neural network
	double networkError = 0.0;
};

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> &myModel)
{
	// This gets the total layers in the network
	unsigned int totalLayers = myModel.size();
	for (unsigned int eachLayer = 0; eachLayer < totalLayers; eachLayer++)
	{
		// Creates the layer
		layers.push_back(Layer());

		// Defines the total outputs from each neuron in the layer
		// Also checks if the layer is the output layer since the output of neurons in the output
		// layer is not fed into any layer. So, it will be zero
		unsigned int totalOutputs = (eachLayer == myModel.size() - 1 ? 0 : myModel[eachLayer + 1]);

		for (unsigned int eachNeuron = 0; eachNeuron <= myModel[eachLayer]; eachNeuron++)
		{
			// Creates the neurons in the layer
			// back() gets the latest element inserted into the vector
			layers.back().push_back(Neuron(totalOutputs, eachNeuron));
		}
	}

	// This sets the output of bias neuron in each layer to 1 
	// except in the output layer since output layer does not have bias neuron
	// Note: But I have considered bias neuron in the output layer as well
	for (unsigned int eachLayer = 0; eachLayer < totalLayers; eachLayer++)
	{
		for (unsigned int eachNeuron = 0; eachNeuron <= myModel[eachLayer]; eachNeuron++)
		{
			if (eachNeuron == myModel[eachLayer])
			{
				layers[eachLayer][eachNeuron].setNeuronOutput(1);
			}
		}
	}
}

void NeuralNetwork::feedForward(const std::vector<double> &inputValues, 
	const std::vector<double> &targetValues)
{
	// Checks whether the input values provided to the network is the same as number of 
	// neurons in the input layer discarding the bias
	assert(inputValues.size() == (layers[0].size() - 1));

	// This sets the output of each neuron in the input layer
	for (unsigned int inputValue = 0; inputValue < inputValues.size(); inputValue++)
	{
		layers[0][inputValue].setNeuronOutput(inputValues[inputValue]);
	}

	// This sets the output of the bias neuron in the input layer to 1
	//layers[0][inputValues.size()].setNeuronOutput(1);

	// Feed forward process
	// Loops through each layer starting from the hidden layer
	for (unsigned int eachLayer = 1; eachLayer < layers.size(); eachLayer++)
	{
		Layer &previousLayer = layers[eachLayer - 1];
		// Loops through each neuron in the layer excluding the bias neuron since the bias neuron
		// produces the constant value
		for (unsigned int eachNeuron = 0; eachNeuron < (layers[eachLayer].size() - 1); eachNeuron++)
		{
			layers[eachLayer][eachNeuron].feedForward(previousLayer, 
				layers[eachLayer][eachNeuron].getNeuronIndex(), layers.size(), eachLayer);
		}
	}

	// This calculates the network error in the feed forward process
	Layer &outputLayer = layers.back();
	networkError = 0.0;
	for (unsigned int eachNeuron = 0; eachNeuron < (outputLayer.size() - 1); eachNeuron++)
	{
		networkError += ((targetValues[eachNeuron] - outputLayer[eachNeuron].getNeuronOutput()) * 
			(inputValues[eachNeuron] - outputLayer[eachNeuron].getNeuronOutput())) / 2;
	}

	std::cout << "Network Error: " << networkError << std::endl;
}

void NeuralNetwork::backPropagation(const std::vector<double> &inputValues, const std::vector<double> &targetValues)
{
	Layer &outputLayer = layers.back();
	// This sets the gradient in the output layer neuron to be used to update weights in the
	// back propagation process
	for (unsigned int eachNeuron = 0; eachNeuron < (outputLayer.size() - 1); eachNeuron++)
	{
		outputLayer[eachNeuron].setGradient((-(targetValues[eachNeuron] - 
			outputLayer[eachNeuron].getNeuronOutput())));
	}

	// This updates the weights of the hidden layer and input layer neurons 
	for (int eachLayer = (layers.size() - 2); eachLayer >= 0; eachLayer--)
	{
		for (unsigned int eachNeuron = 0; eachNeuron < layers[eachLayer].size(); eachNeuron++)
		{
			std::vector<double> &originalWeights = layers[eachLayer][eachNeuron].getWeights();
			double gradient = 0.0;
			for (unsigned int eachWeight = 0; eachWeight < layers[eachLayer][eachNeuron].getWeights().
				size(); eachWeight++)
			{
				double deltaWeight = 0.0;
				deltaWeight = originalWeights[eachWeight] - (0.05 *
					(layers[eachLayer + 1][eachWeight].getGradient() *
					layers[eachLayer][eachNeuron].getNeuronOutput()));
				layers[eachLayer][eachNeuron].updateNeuronWeights(eachWeight, deltaWeight);
				gradient += layers[eachLayer + 1][eachWeight].getGradient() * 
					originalWeights[eachWeight];
			}
			layers[eachLayer][eachNeuron].setGradient(gradient);
		}
	}

	// This transfers the updated weights to the original weights in each neuron in each layer
	// This excludes the output layer since it has no weights associated with it
	// layers.size() - 1 because I have included bias in output layer as well
	// I have to remove bias from the output layer
	for (unsigned int eachLayer = 0; eachLayer < (layers.size() - 1); eachLayer++)
	{
		for (unsigned int eachNeuron = 0; eachNeuron < layers[eachLayer].size(); eachNeuron++)
		{
			std::vector<double> &originalWeights = layers[eachLayer][eachNeuron].getWeights();
			std::vector<double> &updatedWeights = layers[eachLayer][eachNeuron].getUpdatedWeights();
			for (unsigned int eachWeight = 0; eachWeight < originalWeights.size(); eachWeight++)
			{
				layers[eachLayer][eachNeuron].updateOriginalWeights(eachWeight, 
					updatedWeights[eachWeight]);
			}
		}
	}
}


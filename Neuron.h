#pragma once

#include <vector>
#include <cstdlib>
#include <iostream>

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	// Neuron constructor where numOutputs is the number of neurons in the next layer
	// unsigned int because numOutputs cannot be a negative number
	Neuron(unsigned int numOutputs, unsigned int neuronIndex);
	void setNeuronOutput(double neuronInput);
	// This gets the output of each neuron i.e. output after activation
	double getNeuronOutput(void) const;
	// This gets the neuron net output i.e. output before activation
	double getNeuronNetOutput(void) const;
	// This returns the updated weights in back propagation
	std::vector<double> getUpdatedWeights(void) const;
	// This does the feed forward process 
	void feedForward(const Layer &previousLayer, unsigned int neuronIndex, unsigned int networkSize,
		unsigned int layerNum);
	// This gets the neuron index
	unsigned int getNeuronIndex(void) const;
	void updateNeuronWeights(unsigned int index, double deltaWeight);
	// This returns the original weight before back propagation of the neuron
	double getWeight(int neuronIndex) const;
	// This returns the original weight before back propagation of the neuron
	std::vector<double> getWeights(void) const;
	void setGradient(double neuronGradient);
	double getGradient(void) const;
	void updateOriginalWeights(unsigned int index, double deltaWeight);
private:
	// Output value produced by the neuron
	double neuronOutput = 0.0;
	// Weight for sending the output as a weighted output to the neurons in the next layer
	std::vector<double> outputWeight;
	// This contains the updated weight during back propagation for each neuron in each layer
	std::vector<double> updatedWeight;
	// Neuron index in each layer - just for identification in the feed forward and back 
	// propagation processes
	unsigned int nIndex;
	// This is the relu activation funtion - max(0, input)
	static double reluActivationFunction(double input);
	double netOutput = 0.0;
	// This is used to update weights during back propagation
	double gradient = 0.0;
};

Neuron::Neuron(unsigned int numOutputs, unsigned int neuronIndex)
{
	for (unsigned int connections = 0; connections < numOutputs; connections++)
	{
		// Generates a random weight between 0 and 1 for the connections of each neuron
		// in the current layer to the neurons in the next layer
		outputWeight.push_back((rand() / double(RAND_MAX)));
		updatedWeight.push_back(0.0);
	}
	nIndex = neuronIndex;
}

void Neuron::setNeuronOutput(double neuronInput)
{
	neuronOutput = neuronInput;
}

double Neuron::getNeuronOutput(void) const
{
	return neuronOutput;
}

double Neuron::getNeuronNetOutput(void) const
{
	return netOutput;
}

double Neuron::getGradient(void) const
{
	return gradient;
}

unsigned int Neuron::getNeuronIndex(void) const
{
	return nIndex;
}

double Neuron::getWeight(int neuronIndex) const
{
	return outputWeight[neuronIndex];
}

std::vector<double> Neuron::getWeights(void) const
{
	return outputWeight;
}

std::vector<double> Neuron::getUpdatedWeights(void) const
{
	return updatedWeight;
}

double Neuron::reluActivationFunction(double input)
{
	return ((input > 0) ? input : 0);
}

void Neuron::feedForward(const Layer &previousLayer, unsigned int neuronIndex, unsigned int networkSize, 
	unsigned int layerNum)
{
	double sum = 0.0;

	// This loops through the neurons in the previous layer to get its output
	// and multiplies it with the associated weights to calculate the input 
	// for the neurons in the current layer. This includes the bias as well
	for (unsigned int eachNeuron = 0; eachNeuron < previousLayer.size(); eachNeuron++)
	{
		sum += (previousLayer[eachNeuron].getNeuronOutput()) * 
			   (previousLayer[eachNeuron].outputWeight[neuronIndex]);
	}

	netOutput = sum;

	neuronOutput = Neuron::reluActivationFunction(sum);

	if (layerNum == (networkSize - 1))
	{
		std::cout << "Output: " << neuronOutput << std::endl;
	}
}


void Neuron::updateNeuronWeights(unsigned int index, double deltaWeight)
{
	updatedWeight[index] = deltaWeight;
}

void Neuron::updateOriginalWeights(unsigned int index, double updatedWeight)
{
	outputWeight[index] = updatedWeight;
}

void Neuron::setGradient(double neuronGradient)
{
	gradient = neuronGradient;
}